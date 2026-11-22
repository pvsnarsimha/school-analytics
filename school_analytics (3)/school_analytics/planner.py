"""Layer 1 — a rule-based investigator for the AI Assistant. No AI model, no API key, zero cost.

It behaves like a small agent: it starts from a goal (the question), calls a tool, READS the result, and uses fixed rules
to decide what to look at next (drill into the worst section, check the teacher, then the weakest topic ...).
It never repeats a tool call, stops at a step limit, skips checks that do not apply, and shows its steps at the end of the answer.

What it is NOT: it cannot handle questions its rules do not cover, and it does not "reason". For open questions use Claude (see README).
"""
import re
import pandas as pd
import analytics as A

MAX_STEPS = 10
SUBJECT_ALIAS = {"maths": "Mathematics", "math": "Mathematics", "cs": "Computer Science", "computer": "Computer Science",
                 "bio": "Biology", "chem": "Chemistry", "phy": "Physics"}
DIAG_TXT = {"Careless mistakes": "knows the concept but loses easy marks — teach checking habits",
            "Concept gap": "basics are missing — re-teach from the start with simple examples",
            "Needs hard-problem practice": "understands basics — give multi-step problems"}


# ====================================================================== the loop
class Run:
    """One investigation: remembers every step, refuses repeats, enforces the step limit."""

    def __init__(self, T, max_steps=MAX_STEPS):
        self.T, self.max_steps, self.steps, self.tables, self.cache = T, max_steps, [], [], {}

    @property
    def used(self):
        return sum(1 for s in self.steps if not s.get("skipped"))

    def look(self, tool, why, **kw):
        kw = {k: v for k, v in kw.items() if v not in (None, "")}
        key = (tool, tuple(sorted(kw.items())))
        if key in self.cache:
            return self.cache[key]
        if self.used >= self.max_steps:
            self.steps.append({"tool": tool, "args": kw, "why": why, "skipped": "step limit reached"}); return None
        try:
            df = self.T.call(tool, kw)
        except Exception as ex:
            self.steps.append({"tool": tool, "args": kw, "why": why, "skipped": f"not available ({str(ex)[:70]})"}); self.cache[key] = None; return None
        self.steps.append({"tool": tool, "args": kw, "why": why, "found": ""})
        self.tables.append((f"{tool.replace('_', ' ').title()}{' — ' + ', '.join(map(str, kw.values())) if kw else ''}", df))
        self.cache[key] = df
        return df

    def look_many(self, tool, why, argname, values):
        """Same tool for several values (e.g. every subject) — shown as ONE step in the trace, counts as one step."""
        if self.used >= self.max_steps:
            self.steps.append({"tool": tool, "args": {argname: "each"}, "why": why, "skipped": "step limit reached"}); return {}
        out = {}
        for v in values:
            try: out[v] = self.T.call(tool, {argname: v})
            except Exception: pass
        self.steps.append({"tool": tool, "args": {argname: f"each of {len(out)}"}, "why": why, "found": ""})
        return out

    def found(self, text):
        for s in reversed(self.steps):
            if "found" in s:
                s["found"] = text; return

    def skip(self, tool, why, reason):
        self.steps.append({"tool": tool, "args": {}, "why": why, "skipped": reason})

    def trace(self):
        if not self.steps:
            return ""
        out = [f"\n\n🔎 **How I investigated** — {self.used} step{'s' if self.used != 1 else ''}, each chosen by a rule that reacts to the previous result (rule-based, no AI model)"]
        n = 0
        for s in self.steps:
            a = ", ".join(f"{k}={v}" for k, v in s["args"].items())
            if s.get("skipped"):
                out.append(f"- ⏭️ *skipped* `{s['tool']}` — {s['why']} ({s['skipped']})")
            else:
                n += 1; out.append(f"{n}. `{s['tool']}({a})` — {s['why']}" + (f" → **{s['found']}**" if s["found"] else ""))
        return "\n".join(out)


def _after_first_sentence(txt):
    """The tool's plain story starts by repeating the risk %, which the headline already says."""
    return txt.split(". ", 1)[-1]


def _label(f):
    return A.FACTORS.get(f, f).replace(" %", "").lower()


def _compose(run, headline, bullets, actions):
    t = f"**{headline}**\n\n" + "\n".join(f"- {b}" for b in bullets if b)
    if actions:
        t += "\n\n👉 **What to do:** " + " ".join(f"({i}) {a}" for i, a in enumerate(actions, 1)) if len(actions) > 1 else f"\n\n👉 **What to do:** {actions[0]}"
    return t + run.trace(), run.tables


# ====================================================================== playbooks
def subject_playbook(run, subj, sec):
    ss = run.look("subject_summary", "Which subject is weakest, and how bad is it?")
    if subj is None:
        subj = ss.iloc[0].subject
        run.found(f"weakest subject is {subj} (average {ss.iloc[0].average:.0f}, {ss.iloc[0].fail_pct:.0f}% failing)")
    else:
        r = ss[ss.subject == subj].iloc[0]; rank = int(ss.reset_index(drop=True).index[ss.subject.values == subj][0]) + 1
        run.found(f"{subj} average {r.average:.0f} ({r.fail_pct:.0f}% failing), " + (f"the lowest of {len(ss)} subjects" if rank == 1 else f"#{rank} from the bottom of {len(ss)} subjects"))
    r = ss[ss.subject == subj].iloc[0]
    bullets, actions = [], []
    sc = run.look("section_comparison", f"Is {subj} low everywhere, or only in some sections?", subject=subj)
    worst, best = sc.sort_values("average").iloc[0], sc.sort_values("average").iloc[-1]
    gap = best.average - worst.average
    focus = sec or (worst.section if gap > 5 else None)
    if gap > 5:
        run.found(f"Section {worst.section} is {gap:.0f} marks behind Section {best.section} → the problem is concentrated in one class")
        bullets.append(f"**Where:** Section {worst.section} averages {worst.average:.0f} ({worst.fail_pct:.0f}% failing) vs {best.average:.0f} in Section {best.section} — a gap of {gap:.0f} marks.")
        fte = run.look("fair_teacher_effects", "Is it explained by the teacher, after allowing for students' past marks?", subject=subj)
        t = fte[fte.teacher_name == worst.teacher] if fte is not None else pd.DataFrame()
        if len(t):
            t = t.iloc[0]
            if t.verdict in ("Below expected", "Slightly below"):
                run.found(f"{worst.teacher}'s students score {abs(t.effect):.1f} marks below expected (range {t.lo:+.1f} to {t.hi:+.1f})")
                bullets.append(f"**Teacher:** after allowing for students' past marks, attendance and homework, {worst.teacher}'s class scores about {abs(t.effect):.1f} marks below expected (likely range {t.lo:+.1f} to {t.hi:+.1f}).")
                actions.append(f"support {worst.teacher} with a senior mentor, class observation and a lighter timetable — support, not blame")
            else:
                run.found(f"{worst.teacher} is {t.verdict.lower()} → the gap is probably not the teacher")
                bullets.append(f"**Teacher:** after allowing for students' background, {worst.teacher}'s class is {t.verdict.lower()} — so the gap is probably not about the teacher. Look at the students' starting level and class conditions.")
                actions.append(f"check Section {worst.section}'s starting level and timetable (period timing, room, class size)")
    else:
        run.found("all sections are similar → a subject-wide problem, not one class")
        bullets.append(f"**Where:** all sections are within {gap:.0f} marks of each other, so it is the subject itself that students find hard, not one class or teacher.")
    tp = run.look("topic_scores", f"Which {subj} topic is weakest{' in Section ' + focus if focus else ''}?", subject=subj, section=focus)
    if tp is not None and len(tp):
        run.found(f"{tp.iloc[0].topic} ({tp.iloc[0].pct_scored:.0f}% scored)")
        bullets.append(f"**Weakest topic:** {tp.iloc[0].topic} ({tp.iloc[0].pct_scored:.0f}% scored), then {tp.iloc[1].topic} ({tp.iloc[1].pct_scored:.0f}%)." if len(tp) > 1 else f"**Weakest topic:** {tp.iloc[0].topic}.")
        actions.append(f"re-teach {tp.iloc[0].topic} with worked examples and a short check test")
    if gap <= 5:
        mp = run.look("mistake_profile", f"What kind of mistakes do students make in {subj}?", subject=subj)
        if mp is not None and len(mp):
            top = mp.sort_values("students", ascending=False).iloc[0]
            run.found(f"most common: {top.diagnosis} ({top.students} students)")
            bullets.append(f"**Kind of mistakes:** most students show “{top.diagnosis}” ({top.students} students) — {DIAG_TXT.get(top.diagnosis, 'regular practice will help')}.")
    fx = run.look("factor_effects", f"Which student factors are linked with {subj} marks?", subject=subj)
    if fx is not None and len(fx):
        f = fx.iloc[0]
        run.found(f"{_label(f.factor)} ({'+' if f.correlation > 0 else '−'}{abs(f.correlation):.2f})")
        bullets.append(f"**Linked with marks:** {_label(f.factor)} is the factor most linked with {subj} marks ({'higher' if f.correlation > 0 else 'lower'} goes with better marks; linked with — not proof of cause).")
    hw = run.look("homework_summary", f"Is homework a factor in {subj}?", subject=subj)
    if hw is not None and len(hw):
        h = hw.iloc[0]
        run.found(f"{h.avg_submitted_pct:.0f}% submitted, {int(h.students_missing_last3)} students missed the last 3")
        if h.avg_submitted_pct < 80 or h.students_missing_last3 >= 15:
            bullets.append(f"**Homework:** only {h.avg_submitted_pct:.0f}% is submitted and {int(h.students_missing_last3)} students missed their last 3 — worth fixing.")
            actions.append(f"follow up the {int(h.students_missing_last3)} students who missed 3 homeworks")
    head = (f"{subj}: Section {worst.section} is the weakest, averaging {worst.average:.0f} with {worst.fail_pct:.0f}% of its students failing." if gap > 5
            else f"{subj}: weak in every section (average {r.average:.0f}, {r.fail_pct:.0f}% failing).")
    return _compose(run, head, bullets, actions)


def risk_playbook(run, sec):
    ar = run.look("at_risk", f"Who has a 60%+ chance of failing{' in Section ' + sec if sec else ''}?", section=sec)
    if ar is None or not len(ar):
        run.found("nobody at 60%+"); return _compose(run, "Nobody has a 60%+ chance of failing right now.", [], [])
    n = len(ar); by = ar.groupby("section").size().sort_values(ascending=False)
    top = by.index[0]
    run.found(f"{n} students" + (f"; most in Section {top} ({by.iloc[0]})" if not sec and len(by) > 1 else ""))
    bullets = [f"**How many:** {n} student{'s' if n != 1 else ''}{' (list shows the 25 highest)' if n >= 25 else ''}" + (f"; the most are in Section {top} ({by.iloc[0]})." if not sec and len(by) > 1 else ".")]
    ew = run.look("early_warnings", f"How many of them are also absent or missing homework?", section=sec or top)
    ids = set(ar[ar.section == (sec or top)].student_id)
    both = sorted(ids & set(ew.student_id)) if ew is not None else []
    if ew is not None:
        run.found(f"{len(both)} of the {len(ids)} in Section {sec or top} also have an alert")
        if both:
            bullets.append(f"**Most urgent:** {len(both)} in Section {sec or top} are at risk **and** have an attendance/homework alert: {', '.join(both[:6])}{' …' if len(both) > 6 else ''}.")
    hi = ar.sort_values("risk_pct", ascending=False).iloc[0]
    ex = run.look("explain_student", f"Why is the highest-risk student ({hi.student_id}) at risk? (an example of the pattern)", student_id=hi.student_id)
    if ex is not None:
        e = ex.iloc[0]; run.found(e.plain_story[:110] + "…")
        bullets.append(f"**Example ({hi.student_id}, {e.risk_pct:.0f}%):** {_after_first_sentence(e.plain_story)}")
    actions = [f"call the parents of the {len(both)} students with alerts first" if both else "start with the highest-risk students",
               "run remedial classes for all at-risk students", "draft individual plans in AI Agents → Remedial planner"]
    return _compose(run, f"{n} students are likely to fail the final.", bullets, actions)


def absence_playbook(run, sec):
    ew = run.look("early_warnings", "Who is absent for days, or has very low attendance?", section=sec)
    if ew is None or not len(ew):
        run.found("no alerts"); return _compose(run, "No attendance or homework alerts right now.", [], [])
    att = ew[ew.alert.str.startswith(("Absent", "Only"))]; hwk = ew[ew.alert.str.startswith("Missed")]
    run.found(f"{len(att)} attendance alerts, {len(hwk)} homework alerts")
    high = att[att.level == "High"]
    bullets = [f"**Attendance:** {len(att)} student{'s' if len(att) != 1 else ''} flagged ({len(high)} absent 5+ days in a row)." if len(att) else "**Attendance:** no alerts."]
    if len(att) and not sec:
        s = att.groupby("section").size().sort_values(ascending=False); bullets.append(f"**Where:** most in Section {s.index[0]} ({s.iloc[0]}).")
        top = s.index[0]
    else:
        top = sec
    if top and len(att):
        ar = run.look("at_risk", f"Are any of the absent students also at academic risk (Section {top})?", section=top)
        if ar is not None and len(ar):
            overlap = sorted(set(ar.student_id) & set(att.student_id)); run.found(f"{len(overlap)} are also at risk")
            if overlap: bullets.append(f"**Double risk:** {', '.join(overlap[:6])} are absent **and** likely to fail — call these first.")
    hw = run.look("homework_summary", "Which subject has the most students missing homework?")
    if hw is not None and len(hw):
        w = hw.sort_values("students_missing_last3", ascending=False).iloc[0]; run.found(f"{w.subject}: {int(w.students_missing_last3)} students")
        bullets.append(f"**Homework:** {w.subject} has the most students who missed their last 3 homeworks ({int(w.students_missing_last3)}).")
    return _compose(run, f"{len(att)} students have an attendance alert.", bullets,
                    ["class teachers call each parent today (drafts are in AI Agents → Early-warning drafts)", "Principal follows up the 5+ day cases"])


def homework_playbook(run, subj):
    hw = run.look("homework_summary", "How much homework is submitted, by subject?", subject=subj)
    w = hw.sort_values("avg_submitted_pct").iloc[0]
    run.found(f"lowest is {w.subject} ({w.avg_submitted_pct:.0f}% submitted)")
    link = hw.link_with_marks.iloc[0]
    bullets = [f"**Lowest:** {w.subject} — {w.avg_submitted_pct:.0f}% submitted; {int(w.students_missing_last3)} students missed their last 3.",
               f"**Link with marks:** correlation {link:.2f} — {'homework is clearly linked with marks' if link >= .3 else 'only a weak link with marks'} (linked with, not proof of cause)."]
    ew = run.look("early_warnings", "Which students need a call about homework?")
    n = int(ew.alert.str.startswith("Missed").sum()) if ew is not None else 0
    if ew is not None: run.found(f"{n} homework alerts")
    return _compose(run, f"{w.subject} has the lowest homework submission ({w.avg_submitted_pct:.0f}%).", bullets,
                    [f"subject teachers speak to the {n} students who missed 3 in a row" if n else "keep checking weekly", f"HOD checks whether {w.subject} homework is too long or unclear"])


def teacher_playbook(run, subj):
    fte = run.look("fair_teacher_effects", "Which teachers' classes score below what is expected for their students?", subject=subj)
    low = fte[fte.verdict.isin(["Below expected", "Slightly below"])].sort_values("effect")
    if not len(low):
        run.found("nobody differs from expected beyond chance")
        return _compose(run, "No teacher's class scores clearly below expectation after allowing for students' background.",
                        ["Small differences you may see in raw averages are mostly explained by students' past marks, attendance and homework."], ["no action needed; keep supporting every teacher"])
    w = low.iloc[0]; run.found(f"{len(low)} teacher(s) below expected; lowest {w.teacher_name} ({w.effect:+.1f})")
    bullets = [f"**Below expected:** " + "; ".join(f"{r.teacher_name} ({r.subject}) {r.effect:+.1f} marks, range {r.lo:+.1f} to {r.hi:+.1f}" for r in low.head(3).itertuples()) + "."]
    va = run.look("teacher_value_added", f"What are {w.teacher_name}'s experience and workload?", subject=w.subject)
    r = va[va.teacher_name == w.teacher_name]
    if len(r):
        r = r.iloc[0]; extra = f", {int(r.periods_per_week)} periods/week" if "periods_per_week" in r else ""
        run.found(f"{int(r.experience_yrs)} yr{'s' if int(r.experience_yrs) != 1 else ''} experience{extra}"); bullets.append(f"**Context:** {w.teacher_name} has {int(r.experience_yrs)} year{'s' if int(r.experience_yrs) != 1 else ''} of experience{extra}.")
    tp = run.look("topic_scores", f"Which {w.subject} topic is weakest school-wide?", subject=w.subject)
    if tp is not None and len(tp): run.found(f"{tp.iloc[0].topic} ({tp.iloc[0].pct_scored:.0f}%)"); bullets.append(f"**Start with:** {tp.iloc[0].topic}, the weakest {w.subject} topic.")
    return _compose(run, f"{len(low)} teacher{'s' if len(low) != 1 else ''} need{'s' if len(low) == 1 else ''} support (their classes score below expectation).", bullets,
                    ["offer mentoring and class observation, and check workload — this is a support list, not a blame list"])


def mistakes_playbook(run, subj, focus="concept"):
    mp = run.look("mistake_profile", "What kinds of mistakes do students make, by subject?", subject=subj)
    tot = mp.groupby("subject").students.sum(); cg = mp[mp.diagnosis == "Concept gap"].set_index("subject").students.reindex(tot.index).fillna(0)
    share = (cg / tot * 100).sort_values(ascending=False)
    cr = (mp[mp.diagnosis == "Careless mistakes"].set_index("subject").students.reindex(tot.index).fillna(0) / tot * 100).sort_values(ascending=False)
    careless = focus == "careless"
    sj, pct = (cr.index[0], cr.iloc[0]) if careless else (share.index[0], share.iloc[0])
    kind = "careless mistakes" if careless else "concept gaps (basics missing)"
    run.found(f"{sj} has the most {kind} ({pct:.0f}% of students)")
    bullets = [f"**{'Careless mistakes' if careless else 'Concept gaps'}:** highest in {sj} ({pct:.0f}% of students).",
               f"**{'Concept gaps' if careless else 'Careless mistakes'}:** highest in {(share if careless else cr).index[0]} ({(share if careless else cr).iloc[0]:.0f}%)."]
    tp = run.look("topic_scores", f"Which {sj} topic should the work start with?", subject=sj)
    if tp is not None and len(tp): run.found(f"{tp.iloc[0].topic} ({tp.iloc[0].pct_scored:.0f}%)"); bullets.append(f"**Start with:** {tp.iloc[0].topic} in {sj}.")
    small = pct < 10
    head = (f"{kind.capitalize()} are not widespread — highest in {sj}, at {pct:.0f}% of students." if small else f"{sj} has the most students with {kind} ({pct:.0f}%).")
    return _compose(run, head, bullets,
                    ["teach checking habits: re-read the question, show every step, check units" if careless else
                     (f"re-teach the basics of {tp.iloc[0].topic} in {sj}" if tp is not None and len(tp) else f"re-teach the basics in {sj}"),
                     "use the Remedial planner for the individual students affected"])


def trend_playbook(run, subj, sec):
    tr = run.look("trend", "How have marks moved across the year's exams?", subject=subj)
    d = tr.score.iloc[-1] - tr.score.iloc[0]; last = tr.score.iloc[-1] - tr.score.iloc[-2] if len(tr) > 1 else 0
    run.found(f"{'up' if d > 0 else 'down'} {abs(d):.1f} since {tr.exam.iloc[0]}; last exam {last:+.1f}")
    bullets = [f"**Overall:** {'up' if d > 0 else 'down'} {abs(d):.1f} marks from {tr.exam.iloc[0]} ({tr.score.iloc[0]:.0f}) to {tr.exam.iloc[-1]} ({tr.score.iloc[-1]:.0f}); last exam {last:+.1f}.",
               "Exams differ in difficulty — switch on “Adjust for exam difficulty” in the sidebar before judging a drop."]
    actions = []
    if subj is None:
        per = run.look_many("trend", "Which subject moved most? (check every subject)", "subject", sorted(run.T.ss.subject.unique()))
        ch = {k: v.score.iloc[-1] - v.score.iloc[0] for k, v in per.items() if len(v) > 1}
        if ch:
            run.tables.append(("Change in marks by subject", pd.DataFrame({"subject": list(ch), "change": [round(x, 1) for x in ch.values()]}).sort_values("change")))
            wd = min(ch, key=ch.get); bd = max(ch, key=ch.get)
            run.found(f"{wd} fell most ({ch[wd]:+.1f}); {bd} " + (f"rose most ({ch[bd]:+.1f})" if ch[bd] > 0 else f"fell least ({ch[bd]:+.1f})"))
            bullets.append(f"**By subject:** {wd} fell most ({ch[wd]:+.1f}); " + (f"{bd} rose most ({ch[bd]:+.1f})." if ch[bd] > 0 else f"{bd} fell least ({ch[bd]:+.1f}) — no subject rose."))
            if ch[wd] < -2:
                sc = run.look("section_comparison", f"Is the fall in {wd} in one section?", subject=wd)
                if sc is not None:
                    w = sc.sort_values("average").iloc[0]; run.found(f"weakest is Section {w.section} ({w.average:.0f})")
                    bullets.append(f"**Drill-down:** in {wd}, Section {w.section} is lowest ({w.average:.0f}).")
                actions.append(f"look at {wd} first (Section {w.section})" if sc is not None else f"look at {wd} first")
    return _compose(run, f"The school average {'rose' if d > 0 else 'fell'} by {abs(d):.1f} marks.", bullets, actions or ["keep tracking after the next exam"])


def student_playbook(run, sid):
    if sid not in set(run.T.students.student_id):
        return f"I can't show **{sid}** — the ID is not in the students you are allowed to see.", []
    ex = run.look("explain_student", "Why is this student at risk?", student_id=sid)
    if ex is None:
        return f"I can't explain **{sid}** yet — the data needed for the risk explanation is not loaded.{run.trace()}", run.tables
    e = ex.iloc[0]; run.found(f"{e.risk_pct:.0f}% risk")
    att = run.look("attendance_summary", "Is attendance part of the problem?", student_id=sid)
    a = att.iloc[0] if att is not None else None
    streak = int(a.absent_streak) if a is not None and "absent_streak" in a else 0
    low = a is not None and (streak >= 3 or a.get("attendance_last_30_days", 100) < 65 or a.attendance_pct_year < 75)
    bullets = [f"**Why:** {_after_first_sentence(e.plain_story)}"]
    if a is not None:
        run.found(f"{a.attendance_pct_year:.0f}% this year" + (f", absent {streak} days in a row" if streak >= 3 else ""))
        bullets.append(f"**Attendance:** {'absent ' + str(streak) + ' school days in a row; ' if streak >= 3 else ''}{a.attendance_pct_year:.0f}% this year; homework submitted {a.homework_submitted_pct:.0f}%.")
    if low:
        ew = run.look("early_warnings", "Attendance looks weak — does this student have an open alert?")
        mine = ew[ew.student_id == sid] if ew is not None else pd.DataFrame()
        if len(mine): run.found(mine.iloc[0].alert); bullets.append(f"**Alert:** {mine.iloc[0].alert} — {mine.iloc[0].action.lower()}.")
        else: run.found("no open alert for this student")
    else:
        run.skip("early_warnings", "attendance is fine, so no need to check alerts", "rule: attendance ≥ 75%, no streak")
    mp = run.look("mistake_profile", "What kind of mistakes does the student make?", student_id=sid)
    weak = mp[mp.diagnosis.isin(DIAG_TXT)] if mp is not None else pd.DataFrame()
    if mp is not None and not len(weak): run.found("no clear weak pattern (no careless / concept-gap / hard-problem flags)")
    if len(weak):
        run.found("; ".join(f"{r.subject}: {r.diagnosis}" for r in weak.head(3).itertuples()))
        bullets += [f"**{r.subject}:** {r.diagnosis} — {DIAG_TXT[r.diagnosis]}." for r in weak.head(3).itertuples()]
    tg = run.look("student_targets", "What is a realistic next-exam target?", student_id=sid)
    if tg is not None and len(tg):
        w = tg.sort_values("current").iloc[0]; run.found(f"{w.subject}: {w.current:.0f} → {w.target_next_exam:.0f}")
        bullets.append(f"**Target:** weakest subject is {w.subject} (now {w.current:.0f}); a realistic next-exam target is {w.target_next_exam:.0f}.")
    acts = ["call the parent today" if streak >= 3 else None, f"work on {e.can_change_quickly.lower()}" if e.can_change_quickly != "—" else None,
            "start a 2-week remedial plan for the weakest subject (AI Agents → Remedial planner) and re-check after the next exam" if e.risk_pct >= 40 else "keep encouraging; no plan needed"]
    return _compose(run, f"{sid} has about a {e.risk_pct:.0f}% chance of failing at least one subject.", bullets, [x for x in acts if x])


# ====================================================================== router
def _find_subject(options, q):
    ql = q.lower()
    for o in options:
        if o.lower() in ql: return o
    for a, o in SUBJECT_ALIAS.items():
        if re.search(rf"\b{a}\b", ql) and o in options: return o
    return None


def investigate(q, T):
    """Returns (answer_markdown, tables) or None if no playbook fits (the caller then uses its simple fallback)."""
    ql = q.lower(); run = Run(T)
    subj = _find_subject(sorted(T.ss.subject.unique()), q)
    sec = next((s for s in sorted(T.students.section.unique()) if re.search(rf"\bsection\s*{s.lower()}\b|\b{s.lower()}\s*section\b", ql)), None)
    m = re.search(r"\bs\d{4}\b", ql)
    has = lambda *w: any(x in ql for x in w)
    if m: return student_playbook(run, m.group(0).upper())
    if has("risk", "fail the final", "likely to fail", "danger"): return risk_playbook(run, sec)
    if has("absent", "attendance", "early warning", "not coming", "not attending"): return absence_playbook(run, sec)
    if has("homework", "assignment"): return homework_playbook(run, subj)
    if has("careless", "concept gap", "mistake", "basics"): return mistakes_playbook(run, subj, "careless" if "careless" in ql else "concept")
    if has("teacher", "faculty", "who teaches", "mentor"): return teacher_playbook(run, subj)
    if has("improv", "trend", "progress", "better", "worse", "drop", "decline", "fall"): return trend_playbook(run, subj, sec)
    if subj or has("weakest subject", "which subject", "worst subject", "lowest subject", "weak subject"): return subject_playbook(run, subj, sec)
    return None
