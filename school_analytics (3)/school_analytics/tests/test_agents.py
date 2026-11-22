"""Run:  pip install pytest && pytest -q tests      (uses demo_data/, created by make_demo_data.py)"""
import os, sys, json, shutil, datetime as dt, types, subprocess
import pytest, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("ANTHROPIC_API_KEY", None)
import agents as G, ai


@pytest.fixture(scope="module")
def demo_src():
    if not os.path.isdir(os.path.join(ROOT, "demo_data")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "make_demo_data.py")])
    return os.path.join(ROOT, "demo_data")


@pytest.fixture
def d(tmp_path, demo_src):
    dst = tmp_path / "data"; shutil.copytree(demo_src, dst)
    for f in ("agent_queue.csv", "agent_log.csv"):
        (dst / f).unlink(missing_ok=True)
    return G.SchoolData(str(dst), today=dt.date(2026, 11, 22))


def test_briefing_has_three_complete_items(d):
    b = G.weekly_briefing(d)
    assert len(b["items"]) == 3 and b["mode"] == "offline"
    for x in b["items"]:
        assert all(x[k] for k in ("problem", "why", "action", "owner", "deadline"))
    assert "Weekly Principal Briefing" in G.briefing_markdown(b)


def test_action_review_three_decisions_and_nothing_applied_until_approved(d):
    res = G.review_actions(d)
    dec = {r["action_id"]: r["decision"] for r in res["recs"]}
    assert dec[1].startswith("Close") and dec[2].startswith("Extend") and dec[3] == "Change approach" and dec[4] == "Set up measurement"
    assert d.actions().set_index("action_id").status[3] == "In progress"        # unchanged: agent only proposes
    q = G.load_queue(d); item = q[q.title.str.startswith("Action #3")].iloc[0]
    G.decide(d, item.item_id, "Approved", "Principal")
    a = d.actions().set_index("action_id")
    assert a.status[3] == "Dropped" and len(a) == 5 and a.status.iloc[-1] == "Not started"   # follow-up action created


def test_queue_does_not_duplicate_or_nag(d):
    n1 = G.early_warning_agent(d)["items_created"]; n2 = G.early_warning_agent(d)["items_created"]
    assert n1 > 0 and n2 == 0
    q = G.load_queue(d); G.decide(d, q.item_id.iloc[0], "Rejected", "T")
    assert G.early_warning_agent(d)["items_created"] == 0                       # rejected today -> not re-created for 7 days


def test_parent_messages_are_bilingual_and_never_sent(d):
    G.early_warning_agent(d)
    q = G.load_queue(d); r = q[q.kind == "parent_alert"].iloc[0]
    assert "{" not in r.body_en and "{" not in r.body_te                         # all placeholders filled
    assert any("ఀ" <= ch <= "౿" for ch in r.body_te) and r.body_note and r.status == "Pending"
    assert "Call the parent" in r.body_note


def test_data_fix_changes_marks_only_after_approval(d):
    G.data_health_agent(d)
    q = G.load_queue(d); fix = q[q.title.str.startswith("Impossible marks — S1311")].iloc[0]
    assert pd.read_csv(d.path("marks.csv")).query("student_id=='S1311' and exam=='Final' and topic=='Algebra'").marks.iloc[0] == 240
    G.decide(d, fix.item_id, "Approved", "T19")
    assert pd.read_csv(d.path("marks.csv")).query("student_id=='S1311' and exam=='Final' and topic=='Algebra'").marks.iloc[0] == 24
    unsafe = q[q.title.str.startswith("Impossible marks — S1085")].iloc[0]       # no safe guess -> no auto-fix payload
    assert json.loads(unsafe.payload) == {}
    wrong_id = q[q.title.str.contains("S5100")].iloc[0]
    assert "S1500" in wrong_id.body_en


def test_remedial_plan_and_recheck(d):
    p = G.remedial_plan(d, "S1085")
    assert len(p["days"]) == 10 and set(p["worksheets"]) == {"Week 1", "Week 2"} and p["target"] > p["baseline"]
    out = {r["student_id"]: r["outcome"] for r in G.recheck_plans(d)}
    assert out["S1105"] == "Reached target" and out["S1331"] == "Improved" and out["S1479"] == "No improvement"
    n = G.remedial_autoplan(d, 3)
    assert len(n["plans"]) == 3
    item = G.load_queue(d).query("kind=='remedial_plan'").iloc[0]
    G.decide(d, item.item_id, "Approved", "T")
    assert any(x["plan_id"] == json.loads(item.payload)["plan_id"] for x in json.load(open(d.path("remedial_plans.json"), encoding="utf-8")))


def test_roles_only_see_their_items(d):
    G.early_warning_agent(d); G.data_health_agent(d)
    t = G.visible_queue(d, "teacher", "T19"); assert len(t) and set(t.owner) == {"T19"}
    assert len(G.visible_queue(d, "principal", "")) > len(t)
    assert len(G.visible_queue(d, "parent", "S1001")) == 0


# ---------------------------------------------------------------- Claude paths (fake client: no network)
class FakeClient:
    def __init__(self, final_json):
        self.final, self.n, self.messages = final_json, 0, self
    def create(self, **kw):
        self.n += 1
        B = types.SimpleNamespace
        if self.n == 1:
            return B(stop_reason="tool_use", content=[B(type="tool_use", id="1", name="attendance_alerts", input={}), B(type="tool_use", id="2", name="risk_summary", input={})])
        return B(stop_reason="end_turn", content=[B(type="text", text=self.final)])


def _with_claude(monkeypatch, final):
    monkeypatch.setattr(ai, "llm_available", lambda: True)
    monkeypatch.setattr(ai, "_client", lambda: FakeClient(final))


def test_claude_briefing_used_when_grounded(d, monkeypatch):
    ev = G.gather_evidence(d)
    good = json.dumps({"headline": "h", "items": [{"problem": f"{ev['streak3']} students absent 3+ days", "why": f"{ev['high_risk']} students are high risk", "action": "Call parents",
                                                   "owner": "Class teachers", "deadline": "2026-11-23", "evidence": []}]})
    _with_claude(monkeypatch, good)
    b = G.weekly_briefing(d)
    assert b["mode"] == "Claude" and len(b["tool_calls"]) == 2 and b["items"][0]["owner"] == "Class teachers"


def test_claude_briefing_rejected_if_it_invents_a_number(d, monkeypatch):
    bad = json.dumps({"headline": "h", "items": [{"problem": "987 students are absent", "why": "x", "action": "y", "owner": "z", "deadline": "2026-11-23"}]})
    _with_claude(monkeypatch, bad)
    b = G.weekly_briefing(d)
    assert b["mode"] == "offline" and "not in the data" in b["note"] and len(b["items"]) == 3


def test_claude_failure_falls_back_offline(d, monkeypatch):
    monkeypatch.setattr(ai, "llm_available", lambda: True)
    monkeypatch.setattr(ai, "_client", lambda: (_ for _ in ()).throw(RuntimeError("no network")))
    b = G.weekly_briefing(d); assert b["mode"] == "offline" and "unavailable" in b["note"]
    assert G.early_warning_agent(d)["mode"] == "offline"


def test_claude_message_gets_no_student_names(d, monkeypatch):
    seen = []
    monkeypatch.setattr(ai, "llm_available", lambda: True)
    def fake(system, prompt, max_tokens=1500):
        seen.append(prompt)
        return json.dumps({"en": "Dear parent, {STUDENT} missed {N} days. Call {TEACHER}.", "te": "{STUDENT} {N} రోజులు రాలేదు. {TEACHER} గారిని సంప్రదించండి."})
    monkeypatch.setattr(ai, "_complete", fake)
    r = G.early_warning_agent(d)
    assert r["mode"] == "Claude"
    names = " ".join(d.students.student_name.str.split().str[0].unique())
    assert not any(n in " ".join(seen) for n in d.students.student_name.str.split().str[0].unique() if len(n) > 4)
    assert "{" not in G.load_queue(d).body_en.iloc[0]


def test_grounding_check():
    assert G.grounded("12 students, due 2026-11-23", "streak3: 12")
    assert not G.grounded("99 students", "streak3: 12")


# ---------------------------------------------------------------- chat tools
@pytest.fixture
def tools(d):
    def mk(allowed=None):
        ex = dict(daily=d.daily, sections=d.sections, hw_per=d.hw_per, qm=d.qm, qs=d.qs, history=d.history, explain=lambda: d.explained)
        return ai.Tools(d.ss, d.students, d.teachers, d.marks, d.exam, d.risk_prob, allowed_students=allowed, extra=ex)
    return mk


def test_new_tools_work_and_return_no_names(tools):
    T = tools()
    for name, kw in [("early_warnings", {}), ("homework_summary", {}), ("mistake_profile", {"student_id": "S1085"}), ("mistake_profile", {}),
                     ("student_targets", {"student_id": "S1085"}), ("fair_teacher_effects", {}), ("explain_student", {"student_id": "S1085"}), ("attendance_summary", {"student_id": "S1085"})]:
        df = T.call(name, kw); assert len(df), name; assert "student_name" not in df
    ans, tabs = ai._student_answer("S1085", T)
    assert "S1085" in ans and "What to do" in ans and len(tabs) >= 4


def test_tools_respect_role_scope(tools, d):
    mine = d.students[d.students.section == "A"].student_id
    T = tools(allowed=mine)
    other = d.students[d.students.section == "B"].student_id.iloc[0]
    with pytest.raises(ValueError): T.call("explain_student", {"student_id": other})
    ans, _ = ai._student_answer(other, T); assert "can't show" in ans
    assert set(T.early_warnings().student_id) <= set(mine)


def test_chat_offline_routes_student_question(tools):
    ans, tabs, mode = ai.ask_data("Why is S1085 at risk and what should we do?", tools())
    assert mode == "offline" and "chance of failing" in ans


def test_cli_runner(tmp_path, demo_src):
    dst = tmp_path / "data"; shutil.copytree(demo_src, dst)
    for f in ("agent_queue.csv", "agent_log.csv"): (dst / f).unlink(missing_ok=True)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "run_agents.py"), "all", "--data", str(dst), "--no-llm"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert "Drafts added" in out.stdout and any(f.startswith("briefing_") for f in os.listdir(dst / "agent_outputs"))


def test_date_format_is_windows_safe():
    assert G._dstr("2026-11-06") == "6 Nov" and G._dstr("2026-10-30", te=True) == "30 అక్టోబర్"
    assert "%-" not in open(os.path.join(ROOT, "agents.py"), encoding="utf-8").read().replace('"%-d" fails', "")
