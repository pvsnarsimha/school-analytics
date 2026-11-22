"""AI features. Uses Claude (Anthropic API) when ANTHROPIC_API_KEY is set; otherwise a built-in offline mode.

Set the key:  export ANTHROPIC_API_KEY=sk-ant-...   (or add it to .streamlit/secrets.toml)
Optional:     export CLAUDE_MODEL=<model id>     (default below)
"""
import os, json, re, random
import numpy as np, pandas as pd
import analytics as A

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")


def get_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        try:
            import streamlit as st
            k = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            k = None
    return k


def llm_available():
    return bool(get_key())


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=get_key())


def _complete(system, prompt, max_tokens=1500):
    msg = _client().messages.create(model=DEFAULT_MODEL, max_tokens=max_tokens, system=system,
                                    messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text")


# ====================================================================== tools over the analytics
class Tools:
    """Functions the 'Ask the data' agent can call. Every answer is computed from real data, never invented."""

    def __init__(self, ss, students, teachers, marks, exam, risk_prob=None, allowed_students=None, extra=None):
        """extra (optional) unlocks the student-level tools: dict with daily, sections, hw_per, qm, qs, history, attendance, explain (callable)."""
        self.ss, self.students, self.teachers, self.marks, self.exam, self.risk = ss, students, teachers, marks, exam, risk_prob
        self.x = extra or {}
        if allowed_students is not None:   # role scope: a teacher only sees their students
            self.ss = ss[ss.student_id.isin(allowed_students)]
            self.marks = marks[marks.student_id.isin(allowed_students)]
            self.students = students[students.student_id.isin(allowed_students)]
        self.e = self.ss[self.ss.exam == exam].merge(self.students[["student_id", "section"]], on="student_id")

    def subject_summary(self):
        g = self.e.groupby("subject").score.agg(average="mean", fail_pct=lambda x: (x < A.PASS).mean() * 100).round(1).reset_index()
        return g.sort_values("average")

    def section_comparison(self, subject):
        g = self.e[self.e.subject == subject].groupby("section").score.agg(average="mean", fail_pct=lambda x: (x < A.PASS).mean() * 100).round(1).reset_index()
        g["teacher"] = g.section.map(self.e[self.e.subject == subject].groupby("section").teacher_id.agg(lambda x: x.mode().iloc[0])).map(
            self.teachers.set_index("teacher_id").teacher_name)
        return g

    def topic_scores(self, subject, section=None):
        m = self.marks[(self.marks.subject == subject) & (self.marks.exam == self.exam)].merge(self.students[["student_id", "section"]], on="student_id")
        if section: m = m[m.section == section]
        return (m.groupby("topic").marks.mean() / 25 * 100).round(1).rename("pct_scored").reset_index().sort_values("pct_scored")

    def factor_effects(self, subject):
        c = A.factor_correlations(self.ss, self.students, self.exam)
        if subject not in c.columns: return pd.DataFrame()
        r = c[subject].round(2).rename("correlation").reset_index().rename(columns={"index": "factor"})
        return r.reindex(r.correlation.abs().sort_values(ascending=False).index)

    def teacher_value_added(self, subject):
        v = A.value_added(self.ss, self.students, self.teachers, self.exam)
        v = v[v.subject == subject][["teacher_name", "experience_yrs", "avg_score", "value_added"]].round(1)
        if "periods_per_week" in self.teachers:
            v = v.merge(self.teachers[["teacher_name", "periods_per_week"]], on="teacher_name")
        return v

    def trend(self, subject=None):
        s = self.ss if subject is None else self.ss[self.ss.subject == subject]
        return s.groupby(["exam_order", "exam"]).score.mean().round(1).reset_index()[["exam", "score"]]

    def at_risk(self, section=None):
        if self.risk is None: return pd.DataFrame()
        r = pd.DataFrame({"risk_pct": (self.risk * 100).round(0)}).join(self.students.set_index("student_id")[["student_name", "section", "attendance_pct"]], how="inner")
        if section: r = r[r.section == section]
        return r[r.risk_pct >= 60].sort_values("risk_pct", ascending=False).reset_index().head(25)


    # ---------------------------------------------------------------- student-level tools (IDs only — no student names are returned)
    def _sid(self, student_id):
        sid = str(student_id).strip().upper()
        if sid not in set(self.students.student_id):
            raise ValueError(f"{sid} is not a student you can see.")
        return sid

    def _need(self, *keys):
        if any(self.x.get(k) is None for k in keys):
            raise ValueError("This data is not loaded, so this tool is unavailable.")

    def early_warnings(self, section=None):
        self._need("daily", "sections")
        import analytics2 as B
        w = B.early_warnings(self.x["daily"], self.students, self.x["sections"], self.x.get("hw_per"))
        w = w[w.student_id.isin(self.students.student_id)]
        if section: w = w[w.section == section]
        return w[["student_id", "section", "alert", "level", "action"]].head(30)

    def homework_summary(self, subject=None):
        self._need("hw_per")
        import analytics2 as B
        per = self.x["hw_per"]; per = per[per.student_id.isin(self.students.student_id)]
        if subject: per = per[per.subject == subject]
        g = per.groupby("subject").agg(avg_submitted_pct=("submitted_pct", "mean"), students_missing_last3=("missing_last3", lambda x: int((x >= 3).sum()))).round(1).reset_index()
        _, r = B.homework_effect(per, self.ss[self.ss.student_id.isin(self.students.student_id)], self.exam)
        g["link_with_marks"] = round(r, 2)
        return g

    def mistake_profile(self, student_id=None, subject=None):
        self._need("qm", "qs")
        import analytics2 as B
        mp = B.mistake_profile(self.x["qm"], self.x["qs"])
        mp = mp[mp.student_id.isin(self.students.student_id)]
        if student_id: mp = mp[mp.student_id == self._sid(student_id)]
        if subject: mp = mp[mp.subject == subject]
        if student_id: return mp
        return mp.groupby(["subject", "diagnosis"]).size().rename("students").reset_index().sort_values(["subject", "students"], ascending=[True, False])

    def student_targets(self, student_id):
        self._need("history")
        import analytics2 as B
        t = B.student_targets(self.ss, self.x["history"], self._sid(student_id))
        return t[["subject", "current", "best_ever", "trend_per_exam", "target_next_exam"]]

    def fair_teacher_effects(self, subject=None):
        import analytics3 as C3
        out, _ = C3.fair_teacher_effects(self.ss, self.students, self.teachers, self.exam)
        if subject: out = out[out.subject == subject]
        return out[["teacher_name", "subject", "students", "effect", "lo", "hi", "verdict"]].round(1)

    def explain_student(self, student_id):
        self._need("explain")
        import analytics3 as C3
        sid = self._sid(student_id)
        expl, sv, X = self.x["explain"]()
        r = expl.loc[sid]
        return pd.DataFrame([{"student_id": sid, "risk_pct": round(r.risk), "raises_risk": r.raises_risk, "lowers_risk": r.lowers_risk,
                              "can_change_quickly": r.can_change, "plain_story": C3.student_story(sid, expl, sv, X)}])

    def attendance_summary(self, student_id):
        d = self.x.get("daily"); sid = self._sid(student_id)
        row = self.students[self.students.student_id == sid].iloc[0]
        out = {"student_id": sid, "section": row.section, "attendance_pct_year": row.attendance_pct, "homework_submitted_pct": row.assignment_submission_pct}
        if d is not None:
            import analytics2 as B
            s = B.absence_streaks(d).set_index("student_id").loc[sid]
            out.update(absent_streak=int(s.absent_streak), attendance_last_30_days=float(s.attendance_last_30_days))
        return pd.DataFrame([out])

    SPECS = [
        {"name": "subject_summary", "description": "Average score and fail % per subject for the selected exam.", "input_schema": {"type": "object", "properties": {}}},
        {"name": "section_comparison", "description": "Average and fail % per section for one subject, with the teacher of each section.",
         "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]}},
        {"name": "topic_scores", "description": "Percent scored per topic for a subject, optionally for one section.",
         "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "section": {"type": "string"}}, "required": ["subject"]}},
        {"name": "factor_effects", "description": "Correlation of student factors (attendance, study hours, screen time, tuition, internet...) with marks in a subject.",
         "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]}},
        {"name": "teacher_value_added", "description": "For a subject: each teacher's average and marks above/below expectation (value-added), experience and weekly periods.",
         "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]}},
        {"name": "trend", "description": "Average score across the year's exams, optionally for one subject.",
         "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}}}},
        {"name": "at_risk", "description": "Students with 60%+ predicted chance of failing the final, optionally for one section.",
         "input_schema": {"type": "object", "properties": {"section": {"type": "string"}}}},
        {"name": "early_warnings", "description": "Students absent 3+ days in a row, with low 30-day attendance or 3 missed homeworks (IDs only), optionally for one section.",
         "input_schema": {"type": "object", "properties": {"section": {"type": "string"}}}},
        {"name": "homework_summary", "description": "Average homework submission % and number of students who missed the last 3 homeworks, per subject, plus how strongly homework is linked with marks.",
         "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}}}},
        {"name": "mistake_profile", "description": "Question-wise diagnosis (careless mistakes / concept gap / needs hard-problem practice / strong). For one student_id, or the count of students per diagnosis per subject.",
         "input_schema": {"type": "object", "properties": {"student_id": {"type": "string"}, "subject": {"type": "string"}}}},
        {"name": "student_targets", "description": "A realistic next-exam target per subject for one student (from their own trend and best marks).",
         "input_schema": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]}},
        {"name": "fair_teacher_effects", "description": "Fair teacher comparison (multilevel model): marks above/below what is expected for their students, with a likely range. Small classes are shrunk toward average.",
         "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}}}},
        {"name": "explain_student", "description": "Why one student is at risk: risk %, facts raising and lowering it (SHAP), which of them can change quickly, and a plain-language story.",
         "input_schema": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]}},
        {"name": "attendance_summary", "description": "One student's attendance this year, last 30 days, current absence streak, and homework submission %.",
         "input_schema": {"type": "object", "properties": {"student_id": {"type": "string"}}, "required": ["student_id"]}},
    ]

    def call(self, name, args):
        return getattr(self, name)(**{k: v for k, v in (args or {}).items() if v not in (None, "")})


AGENT_SYSTEM = """You are the data assistant for a school's principal and teachers in India.
Answer ONLY from the tool results — never invent numbers. Call tools to get facts first.
Write for a non-technical principal: short, plain English, no jargon (say "marks above expected", not "value-added coefficient").
Structure: 1) direct answer in one sentence, 2) the evidence (2-4 bullet points with numbers), 3) one or two practical actions.
If the data cannot answer the question, say so honestly. Correlation is not proof of cause — say "linked with", not "causes".
For a question about one student (an ID like S1085), combine explain_student, attendance_summary, mistake_profile, student_targets and early_warnings, then give causes and 2-3 practical next steps.
Pass mark is 35. Subjects: {subjects}. Sections: {sections}. Current exam: {exam}."""


def ask_data(question, tools, history=None):
    """Returns (answer_text, tables:list[(title, DataFrame)], mode)."""
    if llm_available():
        try:
            return _ask_llm(question, tools, history or []) + ("Claude",)
        except Exception as ex:
            ans, tabs = _ask_offline(question, tools)
            return f"*(Claude unavailable: {ex.__class__.__name__}. Showing offline answer.)*\n\n" + ans, tabs, "offline"
    return _ask_offline(question, tools) + ("offline",)


def _ask_llm(question, tools, history):
    client = _client()
    sys = AGENT_SYSTEM.format(subjects=", ".join(sorted(tools.ss.subject.unique())), sections=", ".join(sorted(tools.students.section.unique())), exam=tools.exam)
    msgs = [*history[-6:], {"role": "user", "content": question}]
    tables = []
    for _ in range(6):
        r = client.messages.create(model=DEFAULT_MODEL, max_tokens=1200, system=sys, tools=Tools.SPECS, messages=msgs)
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), tables
        msgs.append({"role": "assistant", "content": r.content})
        results = []
        for b in r.content:
            if b.type == "tool_use":
                try:
                    df = tools.call(b.name, b.input)
                    tables.append((f"{b.name.replace('_', ' ').title()} {json.dumps(b.input) if b.input else ''}", df))
                    content = df.drop(columns=[c for c in ("student_name",) if c in df]).to_csv(index=False)  # privacy: no student names to the model
                except Exception as ex:
                    content = f"error: {ex}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content[:6000]})
        msgs.append({"role": "user", "content": results})
    return "I could not finish the analysis — please ask a simpler question.", tables


def _find(options, q):
    ql = q.lower()
    alias = {"maths": "Mathematics", "math": "Mathematics", "cs": "Computer Science", "computer": "Computer Science", "bio": "Biology", "chem": "Chemistry", "phy": "Physics"}
    for o in options:
        if o.lower() in ql: return o
    for a, o in alias.items():
        if re.search(rf"\b{a}\b", ql) and o in options: return o
    return None


def _ask_offline(q, T):
    """Rule-based fallback: understands subject/section names and common question types."""
    subjects = sorted(T.ss.subject.unique()); sections = sorted(T.students.section.unique())
    subj = _find(subjects, q)
    sec = next((s for s in sections if re.search(rf"\bsection\s*{s.lower()}\b|\b{s.lower()}\s*section\b", q.lower())), None)
    ql = q.lower(); tabs = []
    import planner                      # Layer 1: rule-based investigator (multi-step, no AI model). Falls through if no playbook fits.
    r_ = planner.investigate(q, T)
    if r_ is not None:
        return r_
    if any(w in ql for w in ["risk", "fail the final", "likely to fail", "danger"]):
        df = T.at_risk(sec); tabs.append(("Students at high risk", df))
        return (f"**{len(df)} students{' in Section ' + sec if sec else ''} have a 60%+ chance of failing the final.**\n\n"
                f"- Most common signs: low marks in earlier exams, low attendance\n\n👉 Start remedial classes and meet their parents this week."), tabs
    if any(w in ql for w in ["teacher", "faculty", "who teaches"]) and subj:
        df = T.teacher_value_added(subj); tabs.append((f"{subj} teachers", df))
        w = df.sort_values("value_added").iloc[0]
        return (f"**In {subj}, {w.teacher_name}'s students score {abs(w.value_added):.1f} marks {'below' if w.value_added < 0 else 'above'} what we expect from their past results.**\n\n"
                + "\n".join(f"- {r.teacher_name}: average {r.avg_score}, {r.value_added:+.1f} vs expected" for r in df.itertuples())
                + "\n\n👉 Give support (mentor, lighter timetable) rather than blame."), tabs
    if subj and any(w in ql for w in ["topic", "chapter", "lesson"]):
        df = T.topic_scores(subj, sec); tabs.append((f"{subj} topics", df))
        return (f"**Weakest {subj} topic{' in Section ' + sec if sec else ''}: {df.iloc[0].topic} ({df.iloc[0].pct_scored:.0f}% scored).**\n\n"
                + "\n".join(f"- {r.topic}: {r.pct_scored:.0f}%" for r in df.itertuples()) + f"\n\n👉 Re-teach {df.iloc[0].topic} with worked examples and a short test."), tabs
    if subj:
        sc = T.section_comparison(subj); tp = T.topic_scores(subj, sec or sc.sort_values("average").iloc[0].section); fx = T.factor_effects(subj).head(3)
        tabs += [(f"{subj} by section", sc), (f"{subj} topics", tp), (f"What is linked with {subj} marks", fx)]
        worst, best = sc.sort_values("average").iloc[0], sc.sort_values("average").iloc[-1]
        gap = best.average - worst.average
        why = (f"Section {worst.section} ({worst.teacher}) is {gap:.0f} marks behind Section {best.section} — a gap this large in one section usually points to teaching or class conditions."
               if gap > 5 else "All sections are similar, so the subject itself is hard for students.")
        return (f"**{subj}: weakest is Section {worst.section} with average {worst.average:.0f} ({worst.fail_pct:.0f}% failing).**\n\n- {why}\n"
                f"- Weakest topic: {tp.iloc[0].topic} ({tp.iloc[0].pct_scored:.0f}% scored)\n"
                f"- Most linked factors: " + ", ".join(f"{r.factor} ({'+' if r.correlation > 0 else '−'})" for r in fx.itertuples())
                + (f"\n\n👉 Support the Section {worst.section} teacher and re-teach {tp.iloc[0].topic}." if gap > 5 else f"\n\n👉 Re-plan how {tp.iloc[0].topic} is taught in all sections.")), tabs
    if any(w in ql for w in ["improv", "trend", "progress", "better", "worse"]):
        df = T.trend(); tabs.append(("School average across exams", df))
        d = df.score.iloc[-1] - df.score.iloc[0]
        return f"**The school average {'rose' if d > 0 else 'fell'} by {abs(d):.1f} marks from {df.exam.iloc[0]} to {df.exam.iloc[-1]}.**", tabs
    df = T.subject_summary(); tabs.append(("Subject summary", df))
    w = df.iloc[0]
    return (f"**Weakest subject: {w.subject} (average {w.average:.0f}, {w.fail_pct:.0f}% failing).**\n\n"
            + "\n".join(f"- {r.subject}: {r.average:.0f}" for r in df.itertuples())
            + f"\n\nTry asking: *'Why is {w.subject} low?'*, *'Which {w.subject} topic is weakest in Section C?'*, *'Who is at risk in Section B?'*"), tabs


def _student_answer(sid, T):
    """One-student question — handled by the rule-based investigator (planner.py), which skips checks that do not apply."""
    import planner
    return planner.student_playbook(planner.Run(T), sid)


MISTAKE_ACTION_TXT = {"Careless mistakes": "knows the concept but loses easy marks; teach checking habits.", "Concept gap": "basics are missing; re-teach from the start.",
                      "Needs hard-problem practice": "understands basics; give multi-step problems."}


# ====================================================================== teacher feedback
def teacher_feedback(t_name, subject, stats):
    """stats: dict with class_avg, school_avg, value_added, weak_topics, strong_topic, fails, periods, trend."""
    if llm_available():
        try:
            return _complete("You write short, kind, specific feedback for school teachers in India. Start with genuine strengths, "
                             "then 2-3 concrete, practical suggestions. Never blame. Under 170 words. Plain English.",
                             f"Teacher: {t_name}, subject {subject}. Data: {json.dumps(stats, default=str)}"), "Claude"
        except Exception:
            pass
    s = stats
    good = []
    if s["value_added"] > 1: good.append(f"your students scored {s['value_added']:.1f} marks more than expected from their past results")
    if s["class_avg"] >= s["school_avg"]: good.append(f"your class average ({s['class_avg']:.0f}) is above the school average ({s['school_avg']:.0f})")
    good.append(f"your students do best in {s['strong_topic']}")
    sug = [f"Spend 2–3 extra periods on **{s['weak_topics'][0]}** with worked examples, then a 10-minute check test."]
    if len(s["weak_topics"]) > 1: sug.append(f"Add short daily practice on **{s['weak_topics'][1]}**.")
    if s["fails"]: sug.append(f"Meet the {s['fails']} students below the pass mark individually and share a simple weekly plan.")
    if s.get("periods", 0) >= 34: sug.append("Your timetable is very heavy — please talk to the Principal about reducing other duties.")
    return (f"Dear {t_name},\n\nThank you for your work in {subject}. Good news: " + "; ".join(good) + ".\n\nSuggestions for the next exam:\n"
            + "\n".join(f"- {x}" for x in sug) + "\n\nWe are here to support you. — School Academic Team"), "offline"


# ====================================================================== worksheets
OFFLINE_BANK = {
    "Mechanics": ["A car accelerates from rest at 2 m/s² for 5 s. Find its final velocity.", "State Newton's second law and give one daily-life example.",
                  "A 2 kg ball moves at 3 m/s. Find its momentum and kinetic energy."],
    "Electricity": ["Three resistors of 2 Ω, 3 Ω and 6 Ω are in parallel. Find the equivalent resistance.", "A bulb draws 0.5 A from a 220 V supply. Find its power.",
                    "Why are household appliances connected in parallel?"],
    "Optics": ["An object is 20 cm from a convex lens of focal length 10 cm. Find the image distance.", "Draw a ray diagram for a concave mirror with the object beyond C.",
               "Define refractive index. Light speed in glass is 2×10⁸ m/s — find n."],
    "Modern Physics": ["What is the photoelectric effect?", "Find the energy of a photon of wavelength 500 nm (h = 6.6×10⁻³⁴ J s).", "State two properties of alpha particles."],
    "Physical": ["Calculate the molar mass of H₂SO₄.", "How many moles are in 22 g of CO₂?", "Define molarity. Find the molarity of 4 g NaOH in 500 mL."],
    "Organic": ["Write the IUPAC name of CH₃–CH₂–OH.", "Draw the structures of butane and isobutane.", "What is a functional group? Give two examples."],
    "Inorganic": ["Why do metals conduct electricity?", "Write the balanced equation for sodium reacting with water.", "Arrange Na, Mg, Al in order of reactivity."],
    "Practical": ["How will you test for the presence of CO₂ gas?", "Write two safety rules in the chemistry lab.", "Describe how to find the pH of a solution."],
    "Grammar": ["Change into passive voice: 'The teacher praised the students.'", "Correct the error: 'He don't like apples.'", "Fill in with the right tense: 'By next year, she ___ (finish) college.'"],
    "Reading": ["Read a short paragraph and write its main idea in one sentence.", "Find two words in the passage that mean 'happy'.", "Why did the author write this passage? Give one reason."],
    "Writing": ["Write a formal letter to the Principal asking for a library period (120 words).", "Write a notice about a school science fair.", "Write a paragraph on 'My favourite festival' (100 words)."],
    "Literature": ["What is the central theme of the poem you studied last?", "Describe the main character of the story in 5 sentences.", "Explain one figure of speech with an example from the lesson."],
    "Programming": ["Write a Python program to check if a number is even or odd.", "What is the output of: for i in range(3): print(i*2)", "Write a function that returns the largest of three numbers."],
    "Data Structures": ["What is the difference between a stack and a queue?", "Show the stack after: push 3, push 5, pop, push 7.", "Write code to reverse a list without using reverse()."],
    "Databases": ["Write an SQL query to show all students with marks above 60.", "What is a primary key?", "Write SQL to count students in each section."],
    "Networks": ["Expand: LAN, WAN, IP, HTTP.", "What is the difference between a router and a switch?", "Why is HTTPS safer than HTTP?"],
    "Botany": ["Draw and label the parts of a flower.", "What is photosynthesis? Write the equation.", "Differentiate between xylem and phloem."],
    "Zoology": ["Name the five kingdoms of classification.", "What are the features of mammals?", "Explain metamorphosis in a frog."],
    "Genetics": ["Cross a tall (TT) and a dwarf (tt) pea plant. Show the F1 and F2 ratio.", "What is a gene?", "Why does a child resemble its parents?"],
    "Human Physiology": ["Trace the path of blood through the heart.", "What is the function of the kidneys?", "Name the parts of the human digestive system in order."],
}


def _maths_q(topic, rnd, level):
    k = {"Easy": 1, "Medium": 2, "Hard": 3}[level]
    if topic == "Algebra":
        a, b, c = rnd.randint(2, 5 * k), rnd.randint(1, 9 * k), rnd.randint(10, 30 * k)
        x = (c - b) / a
        return f"Solve for x: {a}x + {b} = {c}", f"x = {x:g}"
    if topic == "Trigonometry":
        ang = rnd.choice([30, 45, 60]); f = rnd.choice(["sin", "cos", "tan"])
        val = {("sin", 30): "1/2", ("sin", 45): "1/√2", ("sin", 60): "√3/2", ("cos", 30): "√3/2", ("cos", 45): "1/√2", ("cos", 60): "1/2",
               ("tan", 30): "1/√3", ("tan", 45): "1", ("tan", 60): "√3"}[(f, ang)]
        if k == 1: return f"Find the value of {f} {ang}°.", val
        h = rnd.randint(5, 20) * 2
        return f"A ladder {h} m long makes {ang}° with the ground. How high up the wall does it reach?", f"{h} × sin {ang}° (= {h * np.sin(np.radians(ang)):.2f} m)"
    if topic == "Calculus":
        a, n, b = rnd.randint(1, 4 * k), rnd.randint(2, 2 + k), rnd.randint(1, 9)
        if k == 3:
            return f"Find the maximum or minimum of f(x) = x² − {2 * b}x + {b * b + 3}.", f"Minimum at x = {b}, f({b}) = 3"
        c = lambda v: "" if v == 1 else str(v)
        pw = lambda e: "x" if e == 1 else f"x^{e}"
        return f"Differentiate: f(x) = {c(a)}x^{n} + {c(b)}x", f"f'(x) = {a * n}{pw(n - 1)} + {b}"
    if topic == "Probability":
        r, g = rnd.randint(2, 6 * k), rnd.randint(2, 6 * k)
        if k == 3:
            return f"A bag has {r} red and {g} green balls. Two are drawn without replacement. P(both red)?", f"({r}/{r + g}) × ({r - 1}/{r + g - 1})"
        return f"A bag has {r} red and {g} green balls. One ball is drawn. Find P(red).", f"{r}/{r + g}"
    return f"Practice question on {topic}", ""


def worksheet(subject, topic, level="Medium", n=10, seed=0):
    """Returns (markdown_text, mode)."""
    if llm_available():
        try:
            return _complete("You create school worksheets for Indian Class 9-10 students (CBSE/State Board level). Clear, exam-style, "
                             "numbered questions in increasing difficulty, then an 'Answer key' section with brief answers. Markdown only.",
                             f"Subject: {subject}. Topic: {topic}. Level: {level}. Number of questions: {n}. "
                             f"Students are weak in this topic, so begin with 2 confidence-building questions.", 2500), "Claude"
        except Exception:
            pass
    rnd = random.Random(seed)
    qs, ans = [], []
    if subject == "Mathematics":
        levels = ["Easy"] * 2 + [level] * (n - 2)
        for lv in levels:
            q, a = _maths_q(topic, rnd, lv); qs.append(q); ans.append(a)
    else:
        bank = OFFLINE_BANK.get(topic, [f"Explain one key idea from {topic}."])
        for i in range(n):
            qs.append(bank[i % len(bank)] + ("" if i < len(bank) else " (try a different example)")); ans.append("Teacher to check")
    md = f"## Worksheet — {subject}: {topic} ({level})\n\nName: __________  Section: ____  Date: ______\n\n"
    md += "\n".join(f"{i + 1}. {q}" for i, q in enumerate(qs))
    md += "\n\n---\n### Answer key\n" + "\n".join(f"{i + 1}. {a}" for i, a in enumerate(ans))
    return md, "offline"


# ====================================================================== report card remarks
def remarks(name, scores: dict, attendance, assign, trend, target_sentences, lang="both"):
    """Returns ({'en':..., 'te':...}, mode)."""
    if llm_available():
        try:
            txt = _complete("You write report-card remarks for Indian school students. Warm, honest, specific, 3-4 sentences. "
                            "Return JSON with keys 'en' (English) and 'te' (natural, simple Telugu a parent understands). JSON only.",
                            json.dumps({"name": name, "scores": scores, "attendance": attendance, "assignments": assign, "trend": trend, "targets": target_sentences}), 900)
            d = json.loads(txt[txt.find("{"): txt.rfind("}") + 1]); return d, "Claude"
        except Exception:
            pass
    first = name.split()[0]
    best = max(scores, key=scores.get); worst = min(scores, key=scores.get)
    avg = np.mean(list(scores.values()))
    TEs = {"Mathematics": "గణితం", "Physics": "భౌతిక శాస్త్రం", "Chemistry": "రసాయన శాస్త్రం", "English": "ఆంగ్లం", "Computer Science": "కంప్యూటర్ సైన్స్", "Biology": "జీవ శాస్త్రం"}
    if avg >= 70: en0, te0 = f"{first} is doing very well this term.", f"{first} ఈ టర్మ్‌లో చాలా బాగా చదువుతున్నారు."
    elif avg >= 50: en0, te0 = f"{first} is making steady progress.", f"{first} స్థిరంగా ప్రగతి సాధిస్తున్నారు."
    else: en0, te0 = f"{first} needs more support and regular practice.", f"{first}కి మరింత సహాయం మరియు రోజూ అభ్యాసం అవసరం."
    en = [en0, f"Strongest subject is {best} ({scores[best]:.0f}); {worst} ({scores[worst]:.0f}) needs more attention."]
    te = [te0, f"{TEs.get(best, best)}లో బలంగా ఉన్నారు ({scores[best]:.0f}); {TEs.get(worst, worst)} ({scores[worst]:.0f})పై మరింత శ్రద్ధ అవసరం."]
    if trend > 2: en.append("Marks are improving across exams — well done."); te.append("పరీక్షల వారీగా మార్కులు పెరుగుతున్నాయి — అభినందనలు.")
    elif trend < -2: en.append("Marks have dropped compared with earlier exams."); te.append("ముందటి పరీక్షలతో పోలిస్తే మార్కులు తగ్గాయి.")
    if attendance < 80: en.append(f"Attendance ({attendance:.0f}%) must improve."); te.append(f"హాజరు ({attendance:.0f}%) మెరుగుపడాలి.")
    elif assign < 70: en.append("Please complete homework regularly."); te.append("హోంవర్క్ క్రమం తప్పకుండా పూర్తి చేయాలి.")
    else: en.append("Keep up the regular attendance and homework."); te.append("క్రమమైన హాజరు, హోంవర్క్‌ను ఇలాగే కొనసాగించండి.")
    return {"en": " ".join(en), "te": " ".join(te)}, "offline"
