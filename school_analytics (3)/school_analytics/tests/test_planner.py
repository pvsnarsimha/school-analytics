"""Layer 1: the rule-based investigator (no AI model). Run: pytest -q tests"""
import os, sys, shutil, subprocess, datetime as dt
import pytest, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.pop("ANTHROPIC_API_KEY", None)
import agents as G, ai, planner


@pytest.fixture(scope="module")
def D(tmp_path_factory):
    src = os.path.join(ROOT, "demo_data")
    if not os.path.isdir(src):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "make_demo_data.py")])
    dst = tmp_path_factory.mktemp("d") / "data"; shutil.copytree(src, dst)
    return G.SchoolData(str(dst), today=dt.date(2026, 11, 22))


def make(D, allowed=None, extra=True):
    ex = dict(daily=D.daily, sections=D.sections, hw_per=D.hw_per, qm=D.qm, qs=D.qs, history=D.history, explain=lambda: D.explained) if extra else None
    return ai.Tools(D.ss, D.students, D.teachers, D.marks, D.exam, D.risk_prob, allowed_students=allowed, extra=ex)


QUESTIONS = ["Why are Maths marks low?", "Which subject is weakest?", "Who is at risk in Section B?", "Who is at risk?", "Which students are absent?",
             "Is homework a problem in Physics?", "Which teacher needs support?", "Which subject has careless mistakes?", "Are marks improving?",
             "Why is S1135 at risk and what should we do?", "Why is English low?", "Which Physics topic is weakest in Section C?"]


def steps_of(ans):
    return [l.split("`")[1] for l in ans.splitlines() if l[:2].rstrip(". ").isdigit() and "`" in l]


def test_maths_drills_down_in_order(D):
    ans, tabs = planner.investigate("Why are Maths marks low?", planner.Run(make(D)).T) if False else planner.investigate("Why are Maths marks low?", make(D))
    st = [s.split("(")[0] for s in steps_of(ans)]
    assert st[:4] == ["subject_summary", "section_comparison", "fair_teacher_effects", "topic_scores"]
    assert "Section C" in ans and "How I investigated" in ans and "no AI model" in ans


def test_next_step_depends_on_previous_result(D):
    """If one section is >5 marks behind, check the teacher; if all sections are alike, look at the kind of mistakes instead."""
    T = make(D); seen = set()
    for sj in sorted(D.ss.subject.unique()):
        gap = (lambda sc: sc.average.max() - sc.average.min())(T.section_comparison(sj))
        ans, _ = planner.investigate(f"Why is {sj} low?", T)
        tools = {s.split("(")[0] for s in steps_of(ans)}
        assert ("fair_teacher_effects" in tools) == (gap > 5), (sj, gap, tools)
        assert ("mistake_profile" in tools) == (gap <= 5), (sj, gap, tools)
        seen.add(gap > 5)
    assert seen == {True, False}, "demo data should contain both a concentrated and a subject-wide case"


def test_never_repeats_a_call_and_respects_step_limit(D):
    T = make(D)
    for q in QUESTIONS:
        ans, _ = planner.investigate(q, T)
        calls = [l.split("`")[1] for l in ans.splitlines() if l[:2].rstrip(". ").isdigit() and "`" in l]
        assert len(calls) == len(set(calls)) and len(calls) <= planner.MAX_STEPS, (q, calls)
    r = planner.Run(T, max_steps=2)
    assert r.look("subject_summary", "a") is not None and r.look("trend", "b") is not None
    assert r.look("at_risk", "c") is None and r.steps[-1]["skipped"] == "step limit reached" and r.used == 2


def test_skips_checks_that_do_not_apply(D):
    T = make(D)
    ok = D.students.assign(a=D.students.attendance_pct).query("a>=90").student_id
    sid = next(s for s in ok if T.call("attendance_summary", {"student_id": s}).iloc[0].absent_streak == 0)
    ans, _ = planner.investigate(f"Why is {sid} at risk?", T)
    assert "skipped" in ans and "early_warnings" in ans and "attendance is fine" in ans
    ans2, _ = planner.investigate("Why is S1135 at risk?", T)                      # absent 6 days in a row
    assert "skipped" not in ans2 and "absent 6 school days in a row" in ans2


def test_costs_nothing_never_touches_a_model(D, monkeypatch):
    def boom(*a, **k): raise AssertionError("a model was called")
    monkeypatch.setattr(ai, "_complete", boom); monkeypatch.setattr(ai, "_client", boom)
    T = make(D)
    for q in QUESTIONS + ["What is the capital of France"]:
        ans, tabs, mode = ai.ask_data(q, T)
        assert mode == "offline" and ans


def test_unknown_question_falls_back_to_simple_summary(D):
    assert planner.investigate("What is the capital of France", make(D)) is None
    ans, tabs, mode = ai.ask_data("What is the capital of France", make(D))
    assert "Weakest subject" in ans and mode == "offline"


def test_answer_numbers_match_the_data(D):
    T = make(D)
    ans, _ = planner.investigate("Which subject is weakest?", T)
    w = T.subject_summary().iloc[0]
    assert w.subject in ans.splitlines()[0] and f"{w.average:.0f}" in ans
    ans, _ = planner.investigate("Who is at risk?", T)
    assert f"{len(T.at_risk())} students are likely" in ans


def test_role_scope_and_missing_data(D):
    mine = D.students[D.students.section == "A"].student_id
    T = make(D, allowed=mine)
    other = D.students[D.students.section == "B"].student_id.iloc[0]
    ans, tabs = planner.investigate(f"Why is {other} at risk?", T); assert "can't show" in ans and not tabs
    ans, _ = planner.investigate("Who is at risk?", T)
    assert all(s in set(mine) for t in planner.investigate("Who is at risk?", T)[1] for s in (t[1].get("student_id", []) if "student_id" in t[1] else []))
    T2 = make(D, extra=False)                                                        # student tools unavailable
    ans, _ = planner.investigate("Why is S1135 at risk?", T2); assert "not loaded" in ans
    ans, _ = planner.investigate("Which students are absent?", T2)                  # degrades, does not crash
    assert "alert" in ans.lower()


def test_same_call_is_never_repeated(D):
    r = planner.Run(make(D))
    a = r.look("subject_summary", "first"); b = r.look("subject_summary", "again")
    assert a is b and r.used == 1                                                   # second request is answered from memory, not re-run
    r.look("topic_scores", "x", subject="Physics"); r.look("topic_scores", "y", subject="Physics"); assert r.used == 2
    r.look("topic_scores", "z", subject="Chemistry"); assert r.used == 3           # different arguments = a real new step
