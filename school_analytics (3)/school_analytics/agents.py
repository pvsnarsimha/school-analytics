"""Agentic AI layer — agents that watch the data, decide what matters and prepare the work; people approve.

Agents (each works OFFLINE with rules/templates, and uses Claude when ANTHROPIC_API_KEY is set):
  1. weekly_briefing      "the 3 things to act on this week" (Problem → Why → Action → Owner → Deadline)
  2. review_actions       after each exam: did every open action work? → close / extend / change approach
  3. early_warning_agent  3+ days absent → parent message (English + Telugu) + call-list note for the class teacher
  4. remedial_plan        diagnosis (careless / concept gap …) → 2-week plan + worksheets → re-check next exam
  5. data_health_agent    impossible marks, jumps, unknown IDs → a question + suggested fix for the teacher

Safety rules built in
  • Agents NEVER send messages or change marks by themselves. Everything goes to an approval queue (agent_queue.csv).
  • Every run is written to agent_log.csv (audit trail).
  • Claude never receives student names. Messages are written with {STUDENT} placeholders and filled in locally.
  • Claude's briefing is rejected (offline version used instead) if it quotes a number that is not in the tool results.
"""
import os, re, json, difflib, datetime as dt
from functools import cached_property
import numpy as np, pandas as pd
import analytics as A, analytics2 as B, analytics3 as C3, ai

DEFAULT_DATA = os.environ.get("SCHOOL_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SCHOOL = os.environ.get("SCHOOL_NAME", "School Administration")
HIGH_RISK = 0.6


# ====================================================================== data
class SchoolData:
    """Everything an agent needs, loaded once. Heavy parts (risk model, SHAP) are computed on first use."""

    def __init__(self, data_dir=None, today=None):
        self.dir = data_dir or DEFAULT_DATA
        self.today = today or dt.date.today()
        rd = lambda f: pd.read_csv(self.path(f)) if os.path.exists(self.path(f)) else None
        self.students, self.teachers, self.marks, self.attendance = rd("students.csv"), rd("teachers.csv"), rd("marks.csv"), rd("attendance.csv")
        self.daily, self.sections, self.history = rd("daily_attendance.csv"), rd("sections.csv"), rd("history.csv")
        self.qm, self.qs = rd("question_marks.csv"), rd("questions.csv")
        hw, hws = rd("homework.csv"), rd("homework_status.csv")
        self.hw_per = B.homework_tables(hw, hws)[1] if hw is not None and hws is not None else pd.DataFrame(columns=["student_id", "subject", "submitted_pct", "missing", "missing_last3"])
        self.ss = A.subject_scores(self.marks)
        self.last_order = int(self.ss.exam_order.max())
        self.exam = self.ss[self.ss.exam_order == self.last_order].exam.iloc[0]
        self.tname = self.teachers.set_index("teacher_id").teacher_name.to_dict()
        self.tsubj = self.teachers.set_index("teacher_id").subject.to_dict()
        self.sname = self.students.set_index("student_id").student_name.to_dict()
        self.ssec = self.students.set_index("student_id").section.to_dict()
        if self.sections is None:
            self.sections = pd.DataFrame({"section": self.students.section.unique(), "class_teacher_id": None})
        self.class_teacher = self.sections.set_index("section").class_teacher_id.to_dict()

    def path(self, f):
        return os.path.join(self.dir, f)

    @cached_property
    def risk_prob(self):
        if self.last_order < 2:
            return pd.Series(0.0, index=self.students.student_id)
        return A.risk_model(self.ss, self.students, self.attendance)[0]

    @cached_property
    def explained(self):
        return C3.explain_risk(self.ss, self.students, self.attendance)

    @cached_property
    def targets_tbl(self):
        return B.student_targets(self.ss, self.history) if self.history is not None else pd.DataFrame()

    @cached_property
    def mistakes(self):
        return B.mistake_profile(self.qm, self.qs) if self.qm is not None else pd.DataFrame()

    def actions(self):
        if os.path.exists(self.path("actions.csv")):
            return pd.read_csv(self.path("actions.csv"), dtype={"target_students": str, "deadline": str}).fillna({"target_students": ""})
        return pd.DataFrame(columns=["action_id", "created", "problem", "reason", "action", "owner", "deadline", "status", "subject", "start_exam_order", "target_students"])

    def iso(self, days):
        return (self.today + dt.timedelta(days=days)).isoformat()


# ====================================================================== approval queue + audit log
QUEUE_COLS = ["item_id", "created", "agent", "kind", "level", "student_id", "section", "owner", "title", "body_en", "body_te", "body_note",
              "payload", "key", "status", "decided_by", "decided_at"]
LOG_COLS = ["time", "agent", "mode", "items_created", "summary"]


def load_queue(d):
    if os.path.exists(d.path("agent_queue.csv")):
        return pd.read_csv(d.path("agent_queue.csv"), dtype=str).fillna("").reindex(columns=QUEUE_COLS, fill_value="")
    return pd.DataFrame(columns=QUEUE_COLS)


def save_queue(d, q):
    q.to_csv(d.path("agent_queue.csv"), index=False)


def add_to_queue(d, items):
    """Add items; skip any whose `key` is already waiting, or was decided in the last 7 days (do not nag)."""
    q = load_queue(d)
    recent = d.today - dt.timedelta(days=7)
    blocked = set(q[(q.status == "Pending") | ((q.created >= recent.isoformat()) & q.status.isin(["Approved", "Rejected"]))].key)
    new = []
    for it in items:
        if it["key"] in blocked:
            continue
        blocked.add(it["key"])
        row = {c: "" for c in QUEUE_COLS}
        row.update(it)
        row.update(item_id=f"Q{len(q) + len(new) + 1:04d}", created=d.today.isoformat(), status="Pending",
                   payload=json.dumps(it.get("payload", {}), default=str, ensure_ascii=False))
        new.append(row)
    if new:
        save_queue(d, pd.concat([q, pd.DataFrame(new)], ignore_index=True))
    return len(new)


def log_run(d, agent, mode, n, summary):
    row = pd.DataFrame([{"time": dt.datetime.now().isoformat(timespec="seconds"), "agent": agent, "mode": mode, "items_created": n, "summary": summary}])
    old = pd.read_csv(d.path("agent_log.csv")) if os.path.exists(d.path("agent_log.csv")) else pd.DataFrame(columns=LOG_COLS)
    pd.concat([old, row], ignore_index=True).to_csv(d.path("agent_log.csv"), index=False)


def _apply_action_payload(d, p):
    a = d.actions()
    i = a.index[a.action_id.astype(int) == int(p["action_id"])]
    if len(i):
        if p.get("set_status"): a.loc[i, "status"] = p["set_status"]
        if p.get("set_deadline"): a.loc[i, "deadline"] = p["set_deadline"]
    if p.get("new_action"):
        n = dict(p["new_action"]); n.update(action_id=int(a.action_id.astype(int).max()) + 1, created=d.today.isoformat(), status="Not started")
        a = pd.concat([a, pd.DataFrame([n])], ignore_index=True)
    a.to_csv(d.path("actions.csv"), index=False)


def _apply_marks_fix(d, fx):
    m = pd.read_csv(d.path("marks.csv"))
    sel = (m.student_id == fx["student_id"]) & (m.exam == fx["exam"]) & (m.subject == fx["subject"]) & (m.topic == fx["topic"])
    if sel.sum() != 1:
        raise ValueError("Could not find exactly one matching marks row — nothing changed.")
    m.loc[sel, "marks"] = float(fx["new_marks"])
    m.to_csv(d.path("marks.csv"), index=False)


def _activate_plan(d, plan):
    f = d.path("remedial_plans.json")
    plans = json.load(open(f)) if os.path.exists(f) else []
    plan = dict(plan, status="Active")
    plans = [p for p in plans if p["plan_id"] != plan["plan_id"]] + [plan]
    json.dump(plans, open(f, "w"), ensure_ascii=False, indent=1)


def decide(d, item_id, status, by, body_en=None, body_te=None):
    """A person approves or rejects an item. Approving applies its effect (status change, marks fix, plan start); nothing is ever sent."""
    q = load_queue(d)
    i = q.index[q.item_id == item_id]
    if not len(i):
        raise KeyError(item_id)
    r = q.loc[i[0]]
    if r.status != "Pending":
        return f"{item_id} was already {r.status.lower()}."
    msg = "Rejected."
    if status == "Approved":
        p = json.loads(r.payload or "{}")
        if r.kind == "action_proposal":
            _apply_action_payload(d, p); msg = "Approved — Action Tracker updated."
        elif r.kind == "data_check" and p.get("fix"):
            _apply_marks_fix(d, p["fix"]); msg = f"Approved — marks corrected to {p['fix']['new_marks']}."
        elif r.kind == "remedial_plan":
            _activate_plan(d, p); msg = "Approved — plan is now active and will be re-checked after the next exam."
        elif r.kind == "parent_alert":
            msg = "Approved — copy the message and send it yourself (the system never sends anything)."
        else:
            msg = "Approved — marked as reviewed."
        if body_en is not None: q.loc[i, "body_en"] = body_en
        if body_te is not None: q.loc[i, "body_te"] = body_te
    q.loc[i, ["status", "decided_by", "decided_at"]] = [status, by, dt.datetime.now().isoformat(timespec="seconds")]
    save_queue(d, q)
    log_run(d, "human", "—", 0, f"{by} {status.lower()} {item_id} ({r.kind})")
    return msg


def visible_queue(d, role, scope, status="Pending", kind=None):
    """Which queue items a signed-in user may see."""
    q = load_queue(d)
    if status: q = q[q.status == status]
    if kind: q = q[q.kind.isin([kind] if isinstance(kind, str) else kind)]
    if role == "teacher":
        q = q[q.owner == scope]
    elif role == "hod":
        mine = {t for t, s in d.tsubj.items() if s == scope}
        q = q[q.owner.isin(mine | {f"HOD {scope}"})]
    elif role != "principal":
        q = q.iloc[0:0]
    return q


# ====================================================================== Claude helpers
def _text(r):
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def tool_loop(system, goal, tools, max_steps=8, max_tokens=1800):
    """Generic agent loop: Claude calls tools until it has what it needs. tools = {name: (spec, fn)}.
    Returns (final_text, trace, evidence_text). Raises on API errors (callers fall back to offline mode)."""
    client = ai._client()
    specs = [t[0] for t in tools.values()]
    msgs, trace, evidence = [{"role": "user", "content": goal}], [], []
    for _ in range(max_steps):
        r = client.messages.create(model=ai.DEFAULT_MODEL, max_tokens=max_tokens, system=system, tools=specs, messages=msgs)
        if r.stop_reason != "tool_use":
            return _text(r), trace, "\n".join(evidence)
        msgs.append({"role": "assistant", "content": r.content})
        res = []
        for b in r.content:
            if getattr(b, "type", "") != "tool_use":
                continue
            try:
                out = json.dumps(tools[b.name][1](**(b.input or {})), default=str, ensure_ascii=False)
            except Exception as ex:
                out = f"error: {ex}"
            trace.append({"tool": b.name, "input": b.input}); evidence.append(out)
            res.append({"type": "tool_result", "tool_use_id": b.id, "content": out[:6000]})
        msgs.append({"role": "user", "content": res})
    raise RuntimeError("agent did not finish")


def _json_from(txt):
    return json.loads(txt[txt.find("{"): txt.rfind("}") + 1])


def grounded(text, evidence):
    """True if every number with 2+ digits in `text` also appears in the tool results (dates are ignored)."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    have = set(re.findall(r"\d+(?:\.\d+)?", evidence))
    have |= {str(int(round(float(x)))) for x in have}
    return all(n in have or str(int(round(float(n)))) in have for n in re.findall(r"\d{2,}(?:\.\d+)?", text))


TE_MONTHS = ["జనవరి", "ఫిబ్రవరి", "మార్చి", "ఏప్రిల్", "మే", "జూన్", "జూలై", "ఆగస్టు", "సెప్టెంబర్", "అక్టోబర్", "నవంబర్", "డిసెంబర్"]


def _dstr(x, te=False):
    x = dt.date.fromisoformat(str(x))
    return f"{x.day} {TE_MONTHS[x.month - 1]}" if te else x.strftime("%-d %b")


# ====================================================================== 2. action tracker agent
def review_actions(d, queue=True, owner_filter=None):
    """After an exam: measure every open action against similar students, then recommend close / extend / change approach."""
    recs = []
    last_e = d.ss[d.ss.exam_order == d.last_order]
    for a in d.actions().to_dict("records"):
        if a["status"] in ("Done", "Dropped"):
            continue
        im = C3.matched_impact(d.ss, d.students, a)
        overdue = str(a["deadline"]) < d.today.isoformat()
        r = {"action_id": int(a["action_id"]), "action": a["action"], "problem": a["problem"], "owner": a["owner"], "status": a["status"],
             "deadline": str(a["deadline"]), "overdue": overdue, "subject": a.get("subject") or "", "impact": im}
        prop = {"action_id": int(a["action_id"])}
        ext = d.iso(21)
        if im is None:
            has_t = bool(str(a["target_students"]).strip()) and bool(a.get("subject"))
            if not has_t:
                r.update(decision="Set up measurement", next_step="No target students/subject recorded, so impact cannot be measured. Add them in the Action Tracker.")
            else:
                r.update(decision="Wait for next exam", next_step="Started after the latest exam — result appears once the next exam is entered.")
                if overdue: prop.update(set_deadline=ext); r["next_step"] += f" It is past its deadline: propose extending to {ext}."
        elif im["worked"] and im["effect"] >= 2:
            others = last_e[(last_e.subject == a["subject"]) & (last_e.score < 50) & ~last_e.student_id.isin(str(a["target_students"]).split(";"))].student_id.nunique()
            r.update(decision="Close — it worked", next_step=f"Close it as a success." + (f" {others} other students in {a['subject']} still score below 50 — repeat the same action, starting with the 40 weakest." if others else ""))
            prop.update(set_status="Done")
            if others:
                tgt = last_e[(last_e.subject == a["subject"]) & (last_e.score < 50) & ~last_e.student_id.isin(str(a["target_students"]).split(";"))].sort_values("score").student_id.head(40)
                prop["new_action"] = {"problem": f"{others} more students below 50 in {a['subject']} (start with the 40 weakest)", "reason": f"Action #{a['action_id']} raised marks by about {im['effect']:+.1f}",
                                      "action": a["action"], "owner": a["owner"], "deadline": d.iso(30), "subject": a["subject"],
                                      "start_exam_order": d.last_order, "target_students": ";".join(tgt)}
        elif im["worked"]:
            r.update(decision="Extend — small gain", next_step=f"Real but small gain ({im['effect']:+.1f} marks). Keep it running one more exam and add more practice time; re-check then.")
            prop.update(set_status="In progress", set_deadline=ext)
        elif im["hi"] > 0:
            r.update(decision="Extend — too early to say", next_step=f"Effect {im['effect']:+.1f} marks (range {im['lo']:+.1f} to {im['hi']:+.1f}) includes zero. Keep it running and re-check after the next exam.")
            prop.update(set_status="In progress", set_deadline=ext)
        else:
            diag = _dominant_diagnosis(d, a)
            how = B.MISTAKE_ACTION.get(diag, "Try smaller-group teaching with weekly check tests.") if diag else "Try smaller-group teaching with weekly check tests."
            r.update(decision="Change approach", next_step=f"Targeted students did not gain more than similar students ({im['effect']:+.1f}). " +
                     (f"Most of them show “{diag}”: {how}" if diag else how))
            prop.update(set_status="Dropped")
            prop["new_action"] = {"problem": a["problem"], "reason": f"Action #{a['action_id']} did not help; new approach based on mistake analysis",
                                  "action": how, "owner": a["owner"], "deadline": d.iso(30), "subject": a["subject"],
                                  "start_exam_order": d.last_order, "target_students": a["target_students"]}
        r["proposal"] = prop if len(prop) > 1 else None
        recs.append(r)
    items = []
    for r in recs:
        if r["proposal"] and (owner_filter is None or owner_filter in r["owner"]):
            what = ", ".join(f"{k.replace('set_', '').replace('_', ' ')} → {v}" for k, v in r["proposal"].items() if k != "action_id" and k != "new_action")
            if r["proposal"].get("new_action"): what += "; start a follow-up action"
            items.append({"agent": "Action Tracker agent", "kind": "action_proposal", "level": "Medium", "owner": r["owner"], "title": f"Action #{r['action_id']}: {r['decision']}",
                          "body_en": f"{r['action']}\n\n{r['next_step']}\n\nProposed change: {what}", "payload": r["proposal"],
                          "key": f"action:{r['action_id']}:{r['decision']}"})
    n = add_to_queue(d, items) if queue else 0
    mode = "offline"
    log_run(d, "Action Tracker agent", mode, n, f"Reviewed {len(recs)} open actions; {sum(1 for r in recs if r['decision'].startswith('Close'))} worked, "
            f"{sum(1 for r in recs if r['decision'] == 'Change approach')} need a new approach")
    return {"recs": recs, "items_created": n, "mode": mode}


def _dominant_diagnosis(d, a):
    if d.mistakes is None or not len(d.mistakes) or not a.get("subject"):
        return None
    m = d.mistakes[(d.mistakes.subject == a["subject"]) & d.mistakes.student_id.isin(str(a["target_students"]).split(";"))]
    return m.diagnosis.value_counts().index[0] if len(m) else None


# ====================================================================== 3. early-warning agent
TEMPLATES = {
    "absent": {
        "en": "Dear Parent/Guardian, {STUDENT} of Section {SECTION} has been absent for {N} school days in a row (last in school on {DATE}). "
              "We hope everything is alright. Please let us know the reason and when {STUDENT} will return, or call the class teacher, {TEACHER}. "
              "We are happy to help with any difficulty. — {SCHOOL}",
        "te": "ప్రియమైన తల్లిదండ్రులకు, విద్యార్థి: {STUDENT} (సెక్షన్: {SECTION}) వరుసగా {N} పాఠశాల రోజులుగా హాజరు కాలేదు (చివరిసారి వచ్చిన తేదీ: {DATE}). "
              "అంతా క్షేమంగా ఉందని ఆశిస్తున్నాం. దయచేసి కారణం, మరియు {STUDENT} ఎప్పుడు తిరిగి వస్తారో తెలియజేయండి, లేదా క్లాస్ టీచర్ {TEACHER} గారిని సంప్రదించండి. "
              "ఏదైనా ఇబ్బంది ఉంటే సహాయం చేయడానికి మేము సిద్ధంగా ఉన్నాము. — {SCHOOL}"},
    "low": {
        "en": "Dear Parent/Guardian, {STUDENT} of Section {SECTION} attended only {PCT}% of school days in the last 30 days. "
              "Regular attendance matters a lot for marks. Please help {STUDENT} come every day, and call {TEACHER} if there is any difficulty (health, travel, family). — {SCHOOL}",
        "te": "ప్రియమైన తల్లిదండ్రులకు, విద్యార్థి: {STUDENT} (సెక్షన్: {SECTION}) గత 30 రోజుల్లో కేవలం {PCT}% రోజులు మాత్రమే పాఠశాలకు హాజరయ్యారు. "
              "మార్కులకు క్రమమైన హాజరు చాలా ముఖ్యం. దయచేసి {STUDENT} ప్రతిరోజూ పాఠశాలకు వచ్చేలా చూడండి; ఆరోగ్యం, ప్రయాణం, కుటుంబ సమస్యల వంటి ఇబ్బందులు ఉంటే {TEACHER} గారిని సంప్రదించండి. — {SCHOOL}"},
}


def _fill(txt, v):
    for k, x in v.items():
        txt = txt.replace("{" + k + "}", str(x))
    return txt


def _llm_message(kind, ctx):
    """Claude writes the parent message from facts WITHOUT any name. Returns {'en','te'} with placeholders, or None."""
    system = ("You write short, warm, respectful messages from a school in India to a student's parent. No blame, no threats. 60-90 words. "
              "Use ONLY these placeholders for facts: {STUDENT} {SECTION} {N} {DATE} {PCT} {TEACHER} {SCHOOL}. Never invent facts. "
              "Return JSON only: {\"en\": English message, \"te\": the same message in simple natural Telugu}.")
    txt = ai._complete(system, f"Situation: {kind}. Tone hint: {ctx['tone']}. Facts available as placeholders: " + json.dumps(ctx["facts"]), 700)
    j = _json_from(txt)
    ok = "{STUDENT}" in j["en"] and "{STUDENT}" in j["te"] and re.search(r"[ఀ-౿]", j["te"])
    return j if ok else None


def early_warning_agent(d, use_llm=True, owner_filter=None, min_streak=3, queue=True):
    """Find students absent 3+ days (or very low 30-day attendance), draft parent message (EN+TE) and a call note. A human approves."""
    ew = B.early_warnings(d.daily, d.students, d.sections, d.hw_per, min_streak)
    st = B.absence_streaks(d.daily).set_index("student_id")
    ew = ew[ew.alert.str.startswith(("Absent", "Only"))]
    if owner_filter: ew = ew[ew.class_teacher_id == owner_filter]
    llm_ok, used_llm, items, rows = use_llm and ai.llm_available(), False, [], []
    for r in ew.itertuples():
        s = st.loc[r.student_id]; absent = r.alert.startswith("Absent")
        kind = "absent" if absent else "low"
        v = {"STUDENT": r.student_name.split()[0], "SECTION": r.section, "N": int(s.absent_streak), "PCT": int(s.attendance_last_30_days),
             "DATE": _dstr(s.last_present) if s.last_present != "—" else "—", "TEACHER": d.tname.get(r.class_teacher_id, "the class teacher"), "SCHOOL": SCHOOL}
        risk = float(d.risk_prob.get(r.student_id, 0)) * 100
        tmpl = TEMPLATES[kind]
        if llm_ok:
            try:
                j = _llm_message("absent for several days in a row" if absent else "low attendance over 30 days",
                                 {"tone": "extra supportive: student is also at academic risk" if risk >= 60 else "friendly and caring",
                                  "facts": {k: "(known)" for k in v}})
                if j: tmpl, used_llm = j, True
            except Exception:
                llm_ok = False
        level = r.level
        note = (f"Call the parent {'today' if level == 'High' else 'this week'}. {r.alert}"
                + (f" (last in school {_dstr(s.last_present)})" if absent and s.last_present != "—" else "") + ". "
                "Ask: reason, expected return date, any help needed (health, transport, fees, family). "
                f"Context: attendance this year {d.students.set_index('student_id').attendance_pct[r.student_id]:.0f}%, chance of failing the final {risk:.0f}%. "
                "Be supportive, not accusing." + (" Absent 5+ days: tell the Principal; consider a home visit." if absent and s.absent_streak >= 5 else ""))
        items.append({"agent": "Early-warning agent", "kind": "parent_alert", "level": level, "student_id": r.student_id, "section": r.section,
                      "owner": r.class_teacher_id or "Principal", "title": f"{r.student_name} ({r.student_id}) — {r.alert}",
                      "body_en": _fill(tmpl["en"], v), "body_te": _fill(tmpl["te"], dict(v, DATE=_dstr(s.last_present, te=True) if s.last_present != "—" else "—")), "body_note": note,
                      "payload": {"alert": r.alert, "streak": int(s.absent_streak), "risk_pct": round(risk)}, "key": f"ew:{r.student_id}:{kind}"})
        rows.append({"student_id": r.student_id, "student_name": r.student_name, "section": r.section, "class_teacher": d.tname.get(r.class_teacher_id, "—"),
                     "level": level, "alert": r.alert})
    n = add_to_queue(d, items) if queue else 0
    mode = "Claude" if used_llm else "offline"
    log_run(d, "Early-warning agent", mode, n, f"{len(items)} students flagged; {n} new drafts waiting for approval")
    return {"flagged": pd.DataFrame(rows), "items_created": n, "mode": mode}


# ====================================================================== 4. remedial planner
PLAYBOOK = {
    "Careless mistakes": {
        "focus": "Knows the ideas but loses easy marks — build checking habits.",
        "levels": ("Medium", "Medium"),
        "w1": ["Re-do the last test; label every lost mark “careless” or “didn’t know” (error log)",
               "10 easy {topic} questions, then self-check against the key",
               "Timed set: 8 questions in 15 min using the read → underline → check routine",
               "Teacher reviews the error log; fix the top 2 habits (units, signs, skipped lines)",
               "Friday mini-test (10 marks) on {topic}"],
        "w2": ["Mixed {subject} questions — re-check every answer a second way",
               "Past-paper section on {topic} under exam timing", "Swap papers with a friend and find 3 mistakes",
               "Timed set + update the error log", "Re-check test (15 marks): compare mistakes with week 1"]},
    "Concept gap": {
        "focus": "Basics are missing — re-teach from the start with simple examples.",
        "levels": ("Easy", "Medium"),
        "w1": ["Teacher re-teaches the core idea of {topic} with 2 worked examples",
               "Student explains it back in own words; 6 very easy questions",
               "Prerequisite check: find which earlier idea is missing and fix it",
               "10 easy → medium questions with an answer key, marked with the teacher",
               "Friday 10-minute quiz (basics only)"],
        "w2": ["Re-teach what Friday’s quiz showed is still weak", "12 medium questions similar to the examples",
               "Study with a stronger classmate for 20 minutes", "5 word problems applying {topic}", "Re-check test (15 marks)"]},
    "Needs hard-problem practice": {
        "focus": "Understands the basics — needs practice on multi-step, harder questions.",
        "levels": ("Medium", "Hard"),
        "w1": ["Take 3 multi-step {topic} problems and break each into steps together", "4 multi-step problems with hints available",
               "2 past-exam questions on {topic}: attempt first, then compare with the model answer",
               "Student writes and solves their own multi-step problem", "Friday: 3 harder problems, 30 minutes"],
        "w2": ["Review Friday’s problems — where did the steps break?", "5 hard problems, no hints", "Past-paper hard section under timing",
               "Explain one solved problem to the class (2 min)", "Re-check test (15 marks) incl. 2 hard questions"]},
    "Strong": {
        "focus": "Already strong — stretch with enrichment.", "levels": ("Hard", "Hard"),
        "w1": ["Olympiad-style {topic} problem set (3 questions)", "Read ahead: next {topic} chapter", "Peer-teach a weaker classmate for 15 min", "Challenge set (5 hard)", "Friday reflection"],
        "w2": ["Project: apply {topic} to a real example", "Challenge set (5 hard)", "Past-paper hard section", "Peer-teach", "Re-check test"]},
    "Steady — practise more": {
        "focus": "Steady but not yet secure — regular short practice will lift marks.", "levels": ("Medium", "Medium"),
        "w1": ["Review notes on {topic} (15 min)", "10 application questions", "8 questions, 15-minute timer", "Teacher checks answers and gives feedback", "Friday mini-test (10 marks)"],
        "w2": ["Redo the questions missed in week 1", "12 mixed questions", "Past-paper section on {topic}", "Timed set", "Re-check test (15 marks)"]},
}


def remedial_plan(d, student_id, subject=None):
    """Build a 2-week plan for one student from their mistake diagnosis; includes worksheets and a re-check date."""
    last = d.ss[(d.ss.exam_order == d.last_order) & (d.ss.student_id == student_id)]
    if subject is None:
        subject = last.sort_values("score").iloc[0].subject
    base = float(last[last.subject == subject].score.iloc[0])
    diag = "Steady — practise more"
    if len(d.mistakes):
        m = d.mistakes[(d.mistakes.student_id == student_id) & (d.mistakes.subject == subject)]
        if len(m): diag = m.iloc[0].diagnosis
    tm = d.marks[(d.marks.student_id == student_id) & (d.marks.subject == subject) & (d.marks.exam_order == d.last_order) & (d.marks.absent == 0)]
    topic = (tm.assign(p=tm.marks / tm.max_marks).sort_values("p").iloc[0].topic if len(tm) else "revision")
    target = base + 5
    if len(d.targets_tbl):
        t = d.targets_tbl[(d.targets_tbl.student_id == student_id) & (d.targets_tbl.subject == subject)]
        if len(t): target = max(float(t.iloc[0].target_next_exam), base + 3)
    pb = PLAYBOOK[diag]
    fm = lambda s: s.format(topic=topic, subject=subject)
    days = [{"day": f"Week 1 · Day {i + 1}", "task": fm(x)} for i, x in enumerate(pb["w1"])] + [{"day": f"Week 2 · Day {i + 1}", "task": fm(x)} for i, x in enumerate(pb["w2"])]
    seed = int(re.sub(r"\D", "", student_id) or 0)
    ws = {f"Week {i + 1}": ai.worksheet(subject, topic, lv, 8, seed=seed + i)[0] for i, lv in enumerate(pb["levels"])}
    teacher = last[last.subject == subject].teacher_id.iloc[0]
    return {"plan_id": f"RP-{student_id}-{subject[:3].upper()}-{d.last_order}", "created": d.today.isoformat(), "student_id": student_id,
            "student_name": d.sname.get(student_id, student_id), "section": d.ssec.get(student_id, ""), "subject": subject, "topic": topic, "diagnosis": diag,
            "focus": pb["focus"], "baseline": round(base), "target": round(target), "baseline_exam_order": d.last_order, "recheck_exam_order": d.last_order + 1,
            "owner": teacher, "owner_name": d.tname.get(teacher, teacher), "days": days, "worksheets": ws,
            "success": f"Reach {round(target)} in {subject} in the next exam (now {round(base)})."}


def plan_markdown(p):
    t = f"### Remedial plan — {p['student_name']} ({p['student_id']}), {p['subject']}\n"
    t += f"**Diagnosis:** {p['diagnosis']}  \n**Focus:** {p['focus']}  \n**Weakest topic:** {p['topic']}  \n**Teacher:** {p['owner_name']}  \n**Goal:** {p['success']}\n\n"
    return t + "\n".join(f"- **{x['day']}:** {x['task']}" for x in p["days"])


def remedial_autoplan(d, top_n=10, owner_filter=None, queue=True):
    """After an exam the agent picks the highest-risk students itself, drafts a plan for each and sends them for approval."""
    f = d.path("remedial_plans.json")
    active = {(p["student_id"], p["subject"]) for p in json.load(open(f)) if p["status"] == "Active"} if os.path.exists(f) else set()
    plans = []
    for sid in d.risk_prob[d.risk_prob >= HIGH_RISK].sort_values(ascending=False).index:
        if len(plans) >= top_n: break
        p = remedial_plan(d, sid)
        if (sid, p["subject"]) in active or (owner_filter and p["owner"] != owner_filter): continue
        p["risk_pct"] = round(float(d.risk_prob[sid]) * 100); plans.append(p)
    items = [{"agent": "Remedial planner", "kind": "remedial_plan", "level": "High" if p["risk_pct"] >= 80 else "Medium", "student_id": p["student_id"], "section": p["section"],
              "owner": p["owner"], "title": f"{p['student_name']} ({p['student_id']}) — {p['subject']}: {p['diagnosis']} ({p['risk_pct']}% risk)",
              "body_en": plan_markdown(p), "payload": p, "key": f"plan:{p['plan_id']}"} for p in plans]
    n = add_to_queue(d, items) if queue else 0
    log_run(d, "Remedial planner", "offline", n, f"Planned for the top {len(plans)} at-risk students; {n} plans waiting for approval")
    return {"plans": plans, "items_created": n}


def recheck_plans(d):
    """After the next exam: did each active plan work? Suggest the next step."""
    f = d.path("remedial_plans.json")
    if not os.path.exists(f): return []
    out = []
    for p in json.load(open(f)):
        if p["status"] != "Active" or d.last_order < p["recheck_exam_order"]:
            continue
        r = d.ss[(d.ss.student_id == p["student_id"]) & (d.ss.subject == p["subject"]) & (d.ss.exam_order == p["recheck_exam_order"])]
        if not len(r):
            out.append(dict(p, outcome="Absent", next_step="Student missed the exam — re-test in class.", now=None)); continue
        now = float(r.score.iloc[0]); gain = now - p["baseline"]
        if now >= p["target"]: o, nx = "Reached target", "Close the plan; set a higher target and keep encouraging."
        elif gain >= 3: o, nx = "Improved", "Progress is real but below target — repeat week 2 with fresh worksheets."
        else: o, nx = "No improvement", ("Meet the student and parent; try one-to-one sessions" + ("; re-diagnose — the problem may be a concept gap, not carelessness." if p["diagnosis"] == "Careless mistakes" else "."))
        out.append(dict(p, now=round(now), gain=round(gain, 1), outcome=o, next_step=nx))
    return out


# ====================================================================== 5. data-health agent
def data_health_agent(d, owner_filter=None, queue=True, limit=30):
    """Run the school's data checks and turn each problem into a question + suggested fix for the teacher who entered the marks."""
    checks = A.data_health(d.marks, d.students)
    valid = d.students.student_id.tolist()
    teacher_of = d.marks.drop_duplicates(["student_id", "subject"]).set_index(["student_id", "subject"]).teacher_id.to_dict()
    items, findings = [], []
    imp = checks[0]["rows"]; flagged_pairs = set(zip(imp.student_id, imp.subject)) if len(imp) else set()
    for c in checks:
        findings.append({"check": c["check"], "count": c["count"], "status": c["status"]})
        rows = c["rows"]
        if not c["count"] or rows is None or not len(rows): continue
        if c["check"].startswith("Impossible"):
            for r in rows.itertuples():
                if r.marks >= 100 and r.marks / 10 <= r.max_marks: sug = round(r.marks / 10, 1); why = "an extra digit was typed"
                elif r.marks < 0 and abs(r.marks) <= r.max_marks: sug = abs(r.marks); why = "a minus sign was typed"
                else: sug, why = None, "no safe guess"
                q = f"{r.subject} · {r.topic} · {r.exam}: {r.student_id} has {r.marks:g} out of {r.max_marks:g}."
                if sug is not None: q += f" Did you mean {sug:g}? ({why})"
                else: q += " I cannot guess the right value safely — please check the answer sheet and enter the correct marks."
                items.append({"agent": "Data Health agent", "kind": "data_check", "level": "High", "student_id": r.student_id, "section": d.ssec.get(r.student_id, ""),
                              "owner": r.teacher_id, "title": f"Impossible marks — {r.student_id}, {r.subject}", "body_en": q,
                              "payload": {"fix": {"student_id": r.student_id, "exam": r.exam, "subject": r.subject, "topic": r.topic, "new_marks": sug}} if sug is not None else {},
                              "key": f"dh:imp:{r.student_id}:{r.exam}:{r.subject}:{r.topic}"})
        elif c["check"].startswith("Marks for unknown"):
            for sid in rows.student_id.unique():
                rr = rows[rows.student_id == sid].iloc[0]
                taught = set(d.marks[(d.marks.teacher_id == rr.teacher_id) & (d.marks.subject == rr.subject)].student_id) & set(valid)
                done = set(d.marks[(d.marks.exam == rr.exam) & (d.marks.subject == rr.subject)].student_id)
                missing = sorted(taught - done)   # this teacher's students with no marks yet — the likely owner of the mistyped ID
                close = difflib.get_close_matches(sid, missing, n=1, cutoff=0.5) or difflib.get_close_matches(sid, valid, n=1, cutoff=0.85)
                q = (f"ID {sid} is not in the student list ({rr.subject}, {rr.exam})."
                     + (f" Did you mean {close[0]}? (that student has no {rr.subject} marks for {rr.exam} yet)" if close and close[0] in missing else f" Did you mean {close[0]}?" if close else " Please check the roll number."))
                items.append({"agent": "Data Health agent", "kind": "data_check", "level": "High", "student_id": sid, "owner": rr.teacher_id,
                              "title": f"Unknown student ID — {sid}", "body_en": q, "payload": {}, "key": f"dh:unk:{sid}"})
        elif c["check"].startswith("Sudden"):
            rows = rows[[(a, b) not in flagged_pairs for a, b in zip(rows.student_id, rows.subject)]]
            for r in rows.reindex(rows.change.abs().sort_values(ascending=False).index).head(15).itertuples():
                t = teacher_of.get((r.student_id, r.subject), "Principal")
                items.append({"agent": "Data Health agent", "kind": "data_check", "level": "Medium", "student_id": r.student_id, "section": d.ssec.get(r.student_id, ""), "owner": t,
                              "title": f"Sudden {'drop' if r.change < 0 else 'jump'} — {r.student_id}, {r.subject}",
                              "body_en": f"{r.student_id}'s {r.subject} score changed by {r.change:+.0f} marks between exam {r.from_exam} and {r.to_exam}. "
                                         "Typing slip, wrong row, or genuinely different performance? Please confirm.", "payload": {},
                              "key": f"dh:jump:{r.student_id}:{r.subject}:{r.to_exam}"})
        elif c["check"].startswith("Whole class"):
            for r in rows.head(8).itertuples():
                items.append({"agent": "Data Health agent", "kind": "data_check", "level": "Medium", "section": r.section, "owner": "Principal",
                              "title": f"Identical marks — {r.subject}/{r.topic}, Section {r.section}",
                              "body_en": f"{r.size} students in Section {r.section} have almost identical marks on {r.subject} · {r.topic} ({r.exam}). Copied sheet or pasted column?",
                              "payload": {}, "key": f"dh:same:{r.exam}:{r.subject}:{r.topic}:{r.section}"})
    if owner_filter: items = [i for i in items if i["owner"] == owner_filter]
    items = items[:limit]
    n = add_to_queue(d, items) if queue else 0
    log_run(d, "Data Health agent", "offline", n, f"{sum(1 for f in findings if f['status'] != '🟢')} checks flagged; {n} questions sent to teachers")
    return {"findings": pd.DataFrame(findings), "items_created": n, "mode": "offline"}


# ====================================================================== 1. weekly principal briefing
def gather_evidence(d, actions_review=None):
    st = B.absence_streaks(d.daily).merge(d.students[["student_id", "section"]], on="student_id")
    n3, n5 = int((st.absent_streak >= 3).sum()), int((st.absent_streak >= 5).sum())
    low = int(((st.absent_streak < 3) & (st.attendance_last_30_days < 65)).sum())
    hw = d.hw_per[d.hw_per.missing_last3 >= 3] if len(d.hw_per) else d.hw_per
    hw_sub = hw.groupby("subject").student_id.nunique().sort_values(ascending=False) if len(hw) else pd.Series(dtype=int)
    hi = d.risk_prob[d.risk_prob >= HIGH_RISK]
    hi_sec = pd.Series([d.ssec[s] for s in hi.index]).value_counts() if len(hi) else pd.Series(dtype=int)
    rev = actions_review if actions_review is not None else review_actions(d, queue=False)["recs"]
    health = data_health_agent(d, queue=False)["findings"]
    return {"streak3": n3, "streak5": n5, "streak_by_section": st[st.absent_streak >= 3].section.value_counts().to_dict(), "low_attendance": low,
            "hw_students": int(hw.student_id.nunique()) if len(hw) else 0, "hw_by_subject": hw_sub.to_dict(), "high_risk": int(len(hi)), "high_risk_by_section": hi_sec.to_dict(),
            "overdue_actions": [r for r in rev if r["overdue"]], "failed_actions": [r for r in rev if r["decision"] == "Change approach"],
            "worked_actions": [r for r in rev if r["decision"].startswith("Close")], "open_actions": len(rev),
            "data_flags": int(health[(health.status == "🔴")]["count"].sum()), "problems": A.principal_problems(d.ss, d.students, d.teachers, d.marks, d.exam, d.risk_prob)}


def _candidates(d, ev):
    c = []
    for p in ev["problems"]:
        key = "attendance" if "attendance" in p["problem"] else "risk" if "likely to fail" in p["problem"] else p["problem"]
        c.append({"priority": 50 + 10 * p["severity"], "key": key, "problem": p["problem"], "why": p["reason"], "action": p["action"], "owner": p["owner"],
                  "deadline": p["deadline"], "evidence": [f"{p['subject']}" if p["subject"] != "All" else "School-wide"]})
    if ev["streak3"]:
        secs = ", ".join(f"Section {k} ({v})" for k, v in list(ev["streak_by_section"].items())[:3])
        c.append({"priority": 88 if ev["streak5"] else 66, "key": "attendance", "problem": f"{ev['streak3']} students have been absent 3+ school days in a row ({ev['streak5']} for 5+ days)",
                  "why": f"Long unexplained absences are the fastest route to failing, and the first phone call usually decides whether the child returns. Most affected: {secs}.",
                  "action": "Class teachers call every parent today (use the drafted messages in the approval inbox); Principal contacts the 5+ day cases and decides on home visits",
                  "owner": "Class teachers", "deadline": d.iso(1), "evidence": [f"{ev['streak3']} absent 3+ days", f"{ev['streak5']} absent 5+ days", f"{ev['low_attendance']} more under 65% in 30 days"]})
    if ev["hw_students"]:
        top = list(ev["hw_by_subject"].items())[0]
        c.append({"priority": 52 + min(ev["hw_students"], 40) / 4, "key": "homework", "problem": f"{ev['hw_students']} students missed their last 3 homeworks in a subject",
                  "why": f"Homework is linked with marks in the school's own data; {top[0]} has the most such students ({top[1]}).",
                  "action": f"Subject teachers check the list, speak to each student, and the {top[0]} HOD looks at whether homework is too long or unclear",
                  "owner": f"HOD {top[0]}", "deadline": d.iso(7), "evidence": [f"{ev['hw_students']} students", f"{top[0]}: {top[1]}"]})
    if ev["overdue_actions"]:
        o = ev["overdue_actions"]
        c.append({"priority": 74, "key": "overdue", "problem": f"{len(o)} Action Tracker item{'s are' if len(o) > 1 else ' is'} past the deadline and not marked done",
                  "why": "Actions without follow-up quietly stop; " + "; ".join(f"“{r['action'][:50]}” ({r['owner']})" for r in o[:2]),
                  "action": "Ask each owner for a status today; extend the deadline or close the action (the Action Tracker agent has drafted proposals)", "owner": "Principal",
                  "deadline": d.iso(3), "evidence": [f"{len(o)} overdue of {ev['open_actions']} open"]})
    if ev["failed_actions"]:
        f = ev["failed_actions"][0]
        c.append({"priority": 78, "key": "failed", "problem": f"The action “{f['action'][:60]}” did not raise marks beyond similar students",
                  "why": f["next_step"], "action": "Review in the approval inbox and switch to the proposed new approach", "owner": f["owner"], "deadline": d.iso(7),
                  "evidence": [f"effect {f['impact']['effect']:+.1f} marks"]})
    if ev["data_flags"]:
        c.append({"priority": 90, "key": "data", "problem": f"{ev['data_flags']} marks entries look wrong (impossible values or unknown IDs)",
                  "why": "Wrong marks make every chart, risk score and teacher comparison unreliable.", "action": "Teachers confirm or fix the flagged entries (questions are in the approval inbox)",
                  "owner": "Subject teachers", "deadline": d.iso(2), "evidence": [f"{ev['data_flags']} flagged entries"]})
    return sorted(c, key=lambda x: -x["priority"])


def _pick(cands, n=3):
    seen, out = set(), []
    for x in cands:
        if x["key"] in seen: continue
        seen.add(x["key"]); out.append(x)
    return out[:n], out[n:n + 3]


def _briefing_tools(d, ev):
    sub = {"type": "object", "properties": {}}
    return {
        "top_problems": ({"name": "top_problems", "description": "School's top data-driven problems for the latest exam, each with reason, action, owner, deadline.", "input_schema": sub},
                         lambda: ev["problems"]),
        "attendance_alerts": ({"name": "attendance_alerts", "description": "Counts of students absent 3+/5+ days in a row, by section, and students under 65% in last 30 days.", "input_schema": sub},
                              lambda: {k: ev[k] for k in ("streak3", "streak5", "streak_by_section", "low_attendance")}),
        "homework_alerts": ({"name": "homework_alerts", "description": "Number of students who missed their last 3 homeworks, by subject.", "input_schema": sub},
                            lambda: {"students": ev["hw_students"], "by_subject": ev["hw_by_subject"]}),
        "risk_summary": ({"name": "risk_summary", "description": "Number of students with 60%+ chance of failing the final, by section.", "input_schema": sub},
                         lambda: {"high_risk": ev["high_risk"], "by_section": ev["high_risk_by_section"]}),
        "action_status": ({"name": "action_status", "description": "Open Action Tracker items: overdue, worked, did not help (with measured effect in marks).", "input_schema": sub},
                          lambda: {"open": ev["open_actions"], "overdue": [{"action": r["action"], "owner": r["owner"], "deadline": r["deadline"]} for r in ev["overdue_actions"]],
                                   "worked": [{"action": r["action"], "effect": r["impact"]["effect"]} for r in ev["worked_actions"]],
                                   "did_not_help": [{"action": r["action"], "effect": r["impact"]["effect"]} for r in ev["failed_actions"]]}),
        "data_quality": ({"name": "data_quality", "description": "Number of marks entries flagged as impossible or with unknown student IDs.", "input_schema": sub},
                         lambda: {"flagged_entries": ev["data_flags"]}),
    }


BRIEF_SYSTEM = """You are the weekly briefing agent for the Principal of a school in India. Today is {today}.
First call the tools to gather facts (call every tool that could matter). Then choose the 3 things most worth acting on THIS week,
balancing urgency (children at immediate risk), size and importance.
Rules: use ONLY numbers that appear in tool results. Never mention individual student names. Plain, short English for a non-technical reader.
Return JSON only:
{{"headline": "one sentence", "items": [{{"problem": "...", "why": "...", "action": "...", "owner": "...", "deadline": "YYYY-MM-DD", "evidence": ["...", "..."]}}]}}
with exactly 3 items, most urgent first."""


def weekly_briefing(d, use_llm=True, queue_dependencies=False):
    """The Monday briefing: the 3 things to act on this week, in Problem → Why → Action → Owner → Deadline form."""
    rev = review_actions(d, queue=queue_dependencies)
    ev = gather_evidence(d, rev["recs"])
    cands = _candidates(d, ev)
    top, watch = _pick(cands)
    mode, note, trace = "offline", "", []
    if use_llm and ai.llm_available():
        try:
            txt, trace, evidence = tool_loop(BRIEF_SYSTEM.format(today=d.today), "Prepare this week's briefing.", _briefing_tools(d, ev))
            j = _json_from(txt)
            items = j["items"][:3]
            assert len(items) >= 1 and all(k in x for x in items for k in ("problem", "why", "action", "owner", "deadline"))
            if not grounded(json.dumps(j, ensure_ascii=False), evidence):
                note = "Claude's draft quoted a number that is not in the data, so the built-in version is shown."
            else:
                for x in items: x["evidence"] = x.get("evidence", [])
                top, headline, mode = items, j.get("headline", ""), "Claude"
        except Exception as ex:
            note = f"Claude unavailable ({ex.__class__.__name__}); built-in version shown."
    if mode == "offline":
        headline = f"This week: {top[0]['problem'][0].lower() + top[0]['problem'][1:]}." + (f" Also: {top[1]['problem'][0].lower() + top[1]['problem'][1:]}." if len(top) > 1 else "")
    good = [f"“{r['action'][:70]}” raised marks by about {r['impact']['effect']:+.1f} vs similar students" for r in ev["worked_actions"]]
    pend = len(load_queue(d)[lambda q: q.status == "Pending"])
    tiles = {"High-risk students": ev["high_risk"], "Absent 3+ days": ev["streak3"], "Missed 3 homeworks": ev["hw_students"], "Open actions": ev["open_actions"], "Waiting for approval": pend}
    out = {"date": d.today.isoformat(), "exam": d.exam, "headline": headline, "items": top, "watch": watch, "good_news": good, "tiles": tiles, "mode": mode, "note": note, "tool_calls": trace}
    log_run(d, "Weekly briefing agent", mode, 0, f"Briefing for {d.today.isoformat()}: " + " | ".join(x["problem"][:60] for x in top))
    return out


def briefing_markdown(b):
    t = f"# Weekly Principal Briefing — {b['date']}\n\n**{b['headline']}**\n\n"
    t += " · ".join(f"{k}: **{v}**" for k, v in b["tiles"].items()) + "\n\n"
    for i, x in enumerate(b["items"], 1):
        t += f"## {i}. {x['problem']}\n- **Why:** {x['why']}\n- **Action:** {x['action']}\n- **Owner:** {x['owner']}\n- **Deadline:** {x['deadline']}\n\n"
    if b["good_news"]: t += "## Good news\n" + "\n".join(f"- {g}" for g in b["good_news"]) + "\n\n"
    if b["watch"]: t += "## Also watching\n" + "\n".join(f"- {w['problem']}" for w in b["watch"]) + "\n\n"
    return t + f"_Prepared by the briefing agent ({b['mode']} mode). Numbers come from the school's data; please read before acting._\n"


# ====================================================================== run everything (used by run_agents.py)
def run_all(d, which=("weekly", "after-exam", "daily"), use_llm=True):
    out = {}
    if "daily" in which:
        out["early_warning"] = early_warning_agent(d, use_llm)
        out["data_health"] = data_health_agent(d)
    if "after-exam" in which:
        out["actions"] = review_actions(d)
        out["remedial"] = remedial_autoplan(d)
        out["plan_recheck"] = recheck_plans(d)
    if "weekly" in which:
        out["briefing"] = weekly_briefing(d, use_llm)
    return out
