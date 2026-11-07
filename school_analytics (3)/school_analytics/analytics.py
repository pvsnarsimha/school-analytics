"""Analysis engine: all computations used by the dashboard (NumPy / pandas / SciPy / scikit-learn)."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PASS = 35

FACTORS = {
    "attendance_pct": "Attendance %", "assignment_submission_pct": "Assignment submission %",
    "study_hours_per_day": "Study hrs/day", "sleep_hours": "Sleep hrs", "screen_time_hours": "Screen time hrs",
    "travel_time_min": "Travel time (min)", "lab_attendance_pct": "Lab attendance %",
    "private_tuition": "Private tuition", "home_internet": "Home internet", "previous_year_pct": "Previous year %",
}


def subject_scores(marks):
    """One row per student × exam × subject with score out of 100.
    Absent papers/topics are NOT counted as 0: the score uses only the topics the student sat.
    A paper missed completely is left out (it does not pull averages down)."""
    m = marks.copy()
    if "absent" not in m:
        m["absent"] = 0
    m["sat_marks"] = np.where(m.absent == 1, 0, m.marks)
    m["sat_max"] = np.where(m.absent == 1, 0, m.max_marks)
    g = m.groupby(["student_id", "exam", "exam_order", "subject", "teacher_id"], as_index=False)
    out = g.agg(sat=("sat_marks", "sum"), mx=("sat_max", "sum"))
    out = out[out.mx > 0]
    out["score"] = (out.sat / out.mx * 100).round(1)
    out = out.drop(columns=["sat", "mx"])
    return add_difficulty_adjustment(out)


def add_difficulty_adjustment(ss):
    """'Adjusted' score removes the effect of an exam paper being harder or easier than usual for the whole school.
    For each exam we shift and scale all marks so the school's average and spread match the year's normal level.
    Differences between subjects, sections and students are kept."""
    grand_mean, grand_sd = ss.score.mean(), ss.score.std()
    st_ = ss.groupby("exam_order").score.agg(["mean", "std"])
    mu = ss.exam_order.map(st_["mean"]); sd = ss.exam_order.map(st_["std"])
    ss = ss.copy()
    ss["adj_score"] = ((ss.score - mu) / sd * grand_sd + grand_mean).round(1)
    return ss


def exam_difficulty(ss):
    d = ss.groupby(["exam_order", "exam"]).agg(raw=("score", "mean"), adjusted=("adj_score", "mean")).reset_index()
    d["shift"] = (d.raw - ss.score.mean()).round(1)
    d["verdict"] = np.select([d["shift"] <= -3, d["shift"] >= 3], ["Harder paper than usual", "Easier paper than usual"], "Normal")
    return d


def absence_summary(marks):
    if "absent" not in marks:
        return pd.DataFrame()
    a = marks[marks.absent == 1].groupby(["student_id", "exam", "subject"]).size().reset_index(name="topics_missed")
    return a


def data_health(marks, students):
    """Checks a school should run after every upload. Returns list of dicts: check, status, count, detail, rows."""
    out = []
    m = marks.copy()
    m["pct"] = m.marks / m.max_marks * 100
    bad = m[(m.marks < 0) | (m.marks > m.max_marks)]
    out.append({"check": "Impossible marks (below 0 or above maximum)", "count": len(bad), "status": "🔴" if len(bad) else "🟢", "rows": bad.head(50)})
    unk = m[~m.student_id.isin(students.student_id)]
    out.append({"check": "Marks for unknown student IDs", "count": unk.student_id.nunique(), "status": "🔴" if len(unk) else "🟢", "rows": unk.head(50)})
    ab = absence_summary(m)
    out.append({"check": "Absent entries (kept out of averages, not counted as 0)", "count": len(ab), "status": "🟡" if len(ab) else "🟢", "rows": ab.head(200)})
    ss = subject_scores(m)
    piv = ss.pivot_table(index=["student_id", "subject"], columns="exam_order", values="score")
    jumps = []
    cols = list(piv.columns)
    for a_, b_ in zip(cols[:-1], cols[1:]):
        d = (piv[b_] - piv[a_]).dropna()
        j = d[d.abs() >= 35]
        jumps += [{"student_id": i[0], "subject": i[1], "from_exam": a_, "to_exam": b_, "change": round(v, 1)} for i, v in j.items()]
    jumps = pd.DataFrame(jumps)
    out.append({"check": "Sudden jumps of 35+ marks between exams (check for entry mistakes)", "count": len(jumps), "status": "🟡" if len(jumps) else "🟢", "rows": jumps})
    same = m[m.absent == 0].merge(students[["student_id", "section"]], on="student_id").groupby(["exam", "subject", "topic", "section"]).marks.agg(["nunique", "size"]).reset_index()
    same = same[(same["nunique"] <= 2) & (same["size"] >= 10)]
    out.append({"check": "Whole class with (almost) identical marks on a topic", "count": len(same), "status": "🟡" if len(same) else "🟢", "rows": same})
    missing = students[~students.student_id.isin(m.student_id)]
    out.append({"check": "Students with no marks at all", "count": len(missing), "status": "🟡" if len(missing) else "🟢", "rows": missing.head(50)})
    return out


# ---------------------------------------------------------------- uncertainty & effect size
def mean_ci(x, conf=.95):
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 2:
        return x.mean(), np.nan, np.nan, n
    h = stats.t.ppf((1 + conf) / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), x.mean() - h, x.mean() + h, n


def effect_words(d):
    a = abs(d)
    return "very small" if a < .2 else "small" if a < .5 else "medium" if a < .8 else "large"


def compare_groups(a, b, name_a="Group A", name_b="Group B", unit="marks"):
    """Plain-language comparison: difference, 95% range, effect size (Cohen's d), and confidence."""
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dfree = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    h = stats.t.ppf(.975, dfree) * se
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = diff / sp if sp > 0 else 0
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    lo, hi = diff - h, diff + h
    conf = "High" if (lo > 0 or hi < 0) and min(len(a), len(b)) >= 25 else "Medium" if (lo > 0 or hi < 0) else "Low"
    word = "higher" if diff > 0 else "lower"
    size = effect_words(d)
    sentence = (f"{name_a} is {abs(diff):.1f} {unit} {word} than {name_b} (likely between {min(abs(lo), abs(hi)):.1f} and {max(abs(lo), abs(hi)):.1f}). "
                f"This is a {size} difference." if conf != "Low" else
                f"{name_a} and {name_b} differ by {abs(diff):.1f} {unit}, but this could easily be chance (range {lo:+.1f} to {hi:+.1f}).")
    return {"diff": diff, "lo": lo, "hi": hi, "d": d, "size": size, "p": p, "confidence": conf, "n_a": len(a), "n_b": len(b), "sentence": sentence}


def bh_adjust(p):
    """Benjamini-Hochberg false-discovery-rate correction for many comparisons."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1)
    return out


def section_gaps(ss, students, exam, col="score"):
    """Every section vs the rest of its subject, with range, effect size and many-comparison correction."""
    e = ss[ss.exam == exam].merge(students[["student_id", "section"]], on="student_id")
    rows = []
    for (sj, sec), g in e.groupby(["subject", "section"]):
        rest = e[(e.subject == sj) & (e.section != sec)][col]
        c = compare_groups(g[col], rest, f"Section {sec}", "other sections")
        rows.append({"subject": sj, "section": sec, "average": g[col].mean(), **{k: c[k] for k in ["diff", "lo", "hi", "d", "size", "p", "n_a"]}})
    r = pd.DataFrame(rows)
    r["p_adjusted"] = bh_adjust(r.p.values)
    r["real"] = (r.p_adjusted < .05)
    r["verdict"] = np.where(~r.real, "Normal variation", np.where(r["diff"] < 0, "Really lower", "Really higher"))
    r["worth_acting"] = r.real & (r.d.abs() >= .5)
    return r


def exam_kpis(ss, students, exam):
    e = ss[ss.exam == exam].merge(students[["student_id", "section"]], on="student_id")
    per = e.groupby("student_id").score.agg(["mean", "min"])
    return {"avg": per["mean"].mean(), "pass_rate": (per["min"] >= PASS).mean() * 100,
            "distinction": (per["mean"] >= 75).mean() * 100, "failing": int((per["min"] < PASS).sum()),
            "p90": np.percentile(per["mean"], 90)}


def anova_sections(ss, students, exam):
    """One-way ANOVA: does the section make a real difference per subject?"""
    e = ss[ss.exam == exam].merge(students[["student_id", "section"]], on="student_id")
    rows = []
    for s, g in e.groupby("subject"):
        groups = [x.score.values for _, x in g.groupby("section")]
        f, p = stats.f_oneway(*groups)
        means = g.groupby("section").score.mean()
        rows.append({"Subject": s, "F": round(f, 2), "p-value": p, "Best section": means.idxmax(),
                     "Worst section": means.idxmin(), "Gap (marks)": round(means.max() - means.min(), 1),
                     "Significant?": "Yes" if p < 0.05 else "No"})
    return pd.DataFrame(rows)


def value_added(ss, students, teachers, exam):
    """Teacher value-added: actual score minus score expected from previous-year % (linear fit)."""
    e = ss[ss.exam == exam].merge(students[["student_id", "previous_year_pct"]], on="student_id")
    out = []
    for s, g in e.groupby("subject"):
        b = np.polyfit(g.previous_year_pct, g.score, 1)
        g = g.assign(expected=np.polyval(b, g.previous_year_pct)); g["va"] = g.score - g.expected
        out.append(g)
    e = pd.concat(out)
    t = e.groupby("teacher_id").agg(students=("student_id", "nunique"), avg_score=("score", "mean"),
                                    value_added=("va", "mean"), fail_pct=("score", lambda x: (x < PASS).mean() * 100))
    se = e.groupby("teacher_id").va.sem()
    t["ci95"] = 1.96 * se
    return t.reset_index().merge(teachers, on="teacher_id").sort_values("value_added")


def factor_correlations(ss, students, exam):
    e = ss[ss.exam == exam].pivot_table(index="student_id", columns="subject", values="score")
    d = students.set_index("student_id").loc[e.index]
    return pd.DataFrame({FACTORS[f]: [d[f].corr(e[s]) for s in e.columns] for f in FACTORS},
                        index=e.columns).T


def build_features(ss, students, attendance, upto_exam_order):
    """Features known before the final exam: past exam scores, trend, behaviour factors, attendance trend."""
    past = ss[ss.exam_order <= upto_exam_order]
    piv = past.groupby(["student_id", "exam_order"]).score.mean().unstack()
    f = pd.DataFrame(index=piv.index)
    f["last_exam_avg"] = piv[upto_exam_order]
    f["past_avg"] = piv.mean(1)
    f["score_trend"] = piv[upto_exam_order] - piv[1] if upto_exam_order > 1 else 0
    f["min_subject_last"] = past[past.exam_order == upto_exam_order].groupby("student_id").score.min()
    f["subjects_below_40"] = past[past.exam_order == upto_exam_order].assign(b=lambda x: x.score < 40).groupby("student_id").b.sum()
    a = attendance.copy(); order = {m: i for i, m in enumerate(attendance.month.unique())}
    a["mi"] = a.month.map(order)
    slope = a.groupby("student_id").apply(lambda g: np.polyfit(g.mi, g.attendance_pct, 1)[0], include_groups=False)
    f["attendance_trend"] = slope
    f = f.join(students.set_index("student_id")[list(FACTORS)])
    return f.fillna(0)


FEATURE_LABELS = {"last_exam_avg": "Last exam average", "past_avg": "Average so far", "score_trend": "Score trend",
                  "min_subject_last": "Weakest subject score", "subjects_below_40": "Subjects below 40",
                  "attendance_trend": "Attendance trend (per month)", **FACTORS}


def risk_model(ss, students, attendance):
    """Predict who will fail ≥1 subject (or average < 45) in the FINAL, using data up to the exam before it."""
    last = ss.exam_order.max()
    X = build_features(ss, students, attendance, last - 1)
    fin = ss[ss.exam_order == last].groupby("student_id").score.agg(["min", "mean"])
    y = ((fin["min"] < PASS) | (fin["mean"] < 45)).astype(int).reindex(X.index).fillna(0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
    m = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    pte = m.predict_proba(Xte)[:, 1]
    metrics = {"AUC": roc_auc_score(yte, pte), "Recall": recall_score(yte, pte > .5),
               "Precision": precision_score(yte, pte > .5, zero_division=0), "base_rate": y.mean()}
    m.fit(X, y)  # refit on everything for scoring
    prob = pd.Series(m.predict_proba(X)[:, 1], index=X.index, name="risk")
    imp = pd.Series(m.feature_importances_, index=[FEATURE_LABELS[c] for c in X.columns]).sort_values()
    # reasons: features where the student is in the worst 25% of the school
    bad_low = ["last_exam_avg", "min_subject_last", "attendance_pct", "assignment_submission_pct", "study_hours_per_day", "attendance_trend", "score_trend", "lab_attendance_pct"]
    bad_high = ["screen_time_hours", "subjects_below_40"]
    q = X.quantile([.25, .75])
    def reasons(r):
        out = [FEATURE_LABELS[c] + " low" for c in bad_low if r[c] <= q.loc[.25, c]]
        out += [FEATURE_LABELS[c] + " high" for c in bad_high if r[c] >= q.loc[.75, c] and r[c] > 0]
        return ", ".join(out[:4])
    rs = X.apply(reasons, axis=1).rename("reasons")
    return prob, rs, metrics, imp, X


def risk_band(p):
    return np.select([p >= .6, p >= .3], ["High", "Medium"], "Low")


def cluster_students(ss, students, exam, k=4):
    e = ss[ss.exam == exam].groupby("student_id").score.mean().rename("avg_score")
    d = students.set_index("student_id")[["attendance_pct", "assignment_submission_pct", "study_hours_per_day", "screen_time_hours"]].join(e)
    Z = StandardScaler().fit_transform(d)
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    d["cluster"] = lab
    prof = d.groupby("cluster").mean()
    names = {}
    for c, r in prof.iterrows():
        hi_eff = r.attendance_pct >= d.attendance_pct.median() and r.assignment_submission_pct >= d.assignment_submission_pct.median()
        hi_sc = r.avg_score >= d.avg_score.median()
        names[c] = ("Achievers" if hi_sc and hi_eff else "Struggling despite effort" if hi_eff
                    else "Talented but disengaged" if hi_sc else "Disengaged & low scoring")
    # make names unique if two clusters land on the same label
    seen = {}
    for c in sorted(names, key=lambda c: -prof.loc[c, "avg_score"]):
        n = names[c]; seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1: names[c] = f"{n} ({seen[n]})"
    d["segment"] = d.cluster.map(names)
    return d.reset_index(), prof.assign(segment=prof.index.map(names), size=d.cluster.value_counts().sort_index())


SEGMENT_ACTION = {
    "Achievers": "Enrichment: olympiad / advanced problem sets, peer tutors.",
    "Struggling despite effort": "Teaching help: remedial classes, concept re-teaching, doubt sessions.",
    "Talented but disengaged": "Mentoring & attendance follow-up; parent contact.",
    "Disengaged & low scoring": "Urgent: counselling, parent meeting, daily attendance + remedial.",
}


def topic_matrix(marks, exam):
    t = marks[marks.exam == exam].groupby(["subject", "topic"]).agg(avg=("marks", "mean"), mx=("max_marks", "first"))
    t["pct"] = t.avg / t.mx * 100
    return t.reset_index()


# ======================================================================
# Management-friendly layer: plain language, traffic lights, actions
# ======================================================================
import datetime as _dt

STATUS_ICON = {"Good": "🟢", "Watch": "🟡", "Action needed": "🔴"}


def status_of(avg, fail_pct, target):
    if avg < target - 5 or fail_pct > 8:
        return "Action needed"
    if avg < target or fail_pct > 3:
        return "Watch"
    return "Good"


def subject_status(ss, students, exam, targets, last_year=None):
    order = ss.loc[ss.exam == exam, "exam_order"].iloc[0]
    now = ss[ss.exam == exam].groupby("subject").score.agg(avg="mean", fail=lambda x: (x < PASS).mean() * 100)
    if order > 1:
        prev = ss[ss.exam_order == order - 1].groupby("subject").score.mean()
        now["vs_last_exam"] = now.avg - prev
    else:
        now["vs_last_exam"] = np.nan
    if last_year is not None and order == ss.exam_order.max():
        now["vs_last_year"] = now.avg - last_year.groupby("subject").avg_score.mean()
    else:
        now["vs_last_year"] = np.nan
    now["target"] = [targets["subject_avg"].get(s, 60) for s in now.index]
    now["status"] = [status_of(a, f, t) for a, f, t in zip(now.avg, now.fail, now.target)]
    return now.reset_index().sort_values("avg")


def change_words(x, unit="marks"):
    if pd.isna(x):
        return "—"
    if abs(x) < 0.5:
        return "no change"
    return f"{'▲' if x > 0 else '▼'} {abs(x):.1f} {unit} {'better' if x > 0 else 'worse'}"


def deadline(days):
    return (_dt.date.today() + _dt.timedelta(days=days)).isoformat()


def principal_problems(ss, students, teachers, marks, exam, risk_prob=None):
    """Return the top problems written as Problem → Reason → Action → Owner → Deadline."""
    probs = []
    e = ss[ss.exam == exam].merge(students[["student_id", "section"]], on="student_id")
    tinfo = teachers.set_index("teacher_id")
    # 1. section-subject gap that is statistically real, explained with the teacher
    an = anova_sections(ss, students, exam)
    an = an[an["Significant?"] == "Yes"].sort_values("Gap (marks)", ascending=False)
    if len(an):
        x = an.iloc[0]
        g = e[(e.subject == x.Subject) & (e.section == x["Worst section"])]
        t = g.teacher_id.mode().iloc[0]
        tr = tinfo.loc[t]
        others = e[(e.section == x["Worst section"]) & (e.subject != x.Subject)].score.mean()
        school_others = e[e.subject != x.Subject].score.mean()
        reason = (f"Other subjects in Section {x['Worst section']} are normal ({others:.0f} vs school {school_others:.0f}), "
                  f"so the students are not the problem. The class is taught by {tr.teacher_name} "
                  f"({tr.experience_yrs} yr experience).")
        probs.append({"severity": 3, "subject": x.Subject,
                      "problem": f"{x.Subject} in Section {x['Worst section']} is {x['Gap (marks)']:.0f} marks below Section {x['Best section']}",
                      "reason": reason,
                      "action": f"Assign a senior {x.Subject} mentor to {tr.teacher_name}; weekly class observation; bridge classes for Section {x['Worst section']}",
                      "owner": f"HOD {x.Subject}", "deadline": deadline(14)})
    # 2. hardest topic school-wide (curriculum issue)
    tm = topic_matrix(marks, exam).sort_values("pct")
    h = tm.iloc[0]
    sec_spread = (marks[(marks.exam == exam) & (marks.topic == h.topic)].merge(students[["student_id", "section"]], on="student_id")
                  .groupby("section").marks.mean() / h.mx * 100)
    probs.append({"severity": 2, "subject": h.subject,
                  "problem": f"Students score only {h.pct:.0f}% in {h.topic} ({h.subject})",
                  "reason": f"It is weak in every section ({sec_spread.min():.0f}–{sec_spread.max():.0f}%), so it is a topic/teaching-method problem, not one teacher.",
                  "action": f"Re-plan {h.topic}: extra periods, worked examples, a revision test before the next exam",
                  "owner": f"HOD {h.subject}", "deadline": deadline(21)})
    # 3. attendance
    low = students[students.attendance_pct < 75]
    if len(low):
        avg_low = e[e.student_id.isin(low.student_id)].groupby("student_id").score.mean().mean()
        avg_hi = e[~e.student_id.isin(low.student_id)].groupby("student_id").score.mean().mean()
        probs.append({"severity": 2, "subject": "All",
                      "problem": f"{len(low)} students have attendance below 75%",
                      "reason": f"They score {avg_hi - avg_low:.0f} marks less on average than regular students.",
                      "action": "Call parents of every student below 75%; monthly attendance review in staff meeting",
                      "owner": "Class teachers", "deadline": deadline(7)})
    # 4. at-risk students
    if risk_prob is not None:
        n = int((risk_prob >= .6).sum())
        if n:
            probs.append({"severity": 3 if n > 40 else 2, "subject": "All",
                          "problem": f"{n} students are likely to fail the final exam",
                          "reason": "Low marks in earlier exams plus low attendance / assignment submission.",
                          "action": "Remedial classes + parent meeting for every high-risk student",
                          "owner": "Principal", "deadline": deadline(10)})
    return sorted(probs, key=lambda p: -p["severity"])


def headline(status_df, problems, kpi, kpi_prev):
    red = status_df[status_df.status == "Action needed"].subject.tolist()
    trend = ""
    if kpi_prev:
        d = kpi["avg"] - kpi_prev["avg"]
        trend = "Overall results are stable" if abs(d) < 1 else ("Overall results improved" if d > 0 else "Overall results dropped")
    else:
        trend = "First exam of the year"
    focus = problems[0]["problem"] if problems else "no urgent problems"
    red_txt = f"{', '.join(red)} need{'s' if len(red) == 1 else ''} attention" if red else "all subjects on track"
    return f"{trend}; {red_txt}. Most urgent: {focus}."


def target_progress(ss, students, exam, targets):
    k = exam_kpis(ss, students, exam)
    rows = [("Pass rate (all subjects)", k["pass_rate"], targets["pass_rate"], "%"),
            ("School average", k["avg"], targets["school_avg"], ""),
            ("Distinction rate", k["distinction"], targets["distinction"], "%"),
            ("Average attendance", students.attendance_pct.mean(), targets["attendance"], "%")]
    return pd.DataFrame(rows, columns=["Goal", "Now", "Target", "unit"]).assign(
        progress=lambda d: (d.Now / d.Target * 100).clip(upper=100))


def action_impact(ss, action):
    """Did the action work? Compare target group's change with everyone else's, same subject."""
    ids = [x for x in str(action.get("target_students", "")).split(";") if x]
    subj, start = action.get("subject"), int(action.get("start_exam_order", 0) or 0)
    last = ss.exam_order.max()
    if not ids or not subj or subj == "All" or start == 0 or start >= last:
        return None
    s = ss[ss.subject == subj].pivot_table(index="student_id", columns="exam_order", values="score")
    ch = s[last] - s[start]
    grp, oth = ch[ch.index.isin(ids)], ch[~ch.index.isin(ids)]
    t, p = stats.ttest_ind(grp, oth, equal_var=False)
    return {"n": len(grp), "group_change": grp.mean(), "others_change": oth.mean(),
            "net": grp.mean() - oth.mean(), "p": p, "worked": grp.mean() - oth.mean() > 1 and p < .05,
            "before": s.loc[s.index.isin(ids), start].mean(), "after": s.loc[s.index.isin(ids), last].mean()}


def validate_marks_sheet(df, students, topics, max_marks=25):
    """Check a teacher's filled Excel template. Returns (errors, warnings, clean_long_df)."""
    errors, warns = [], []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "student_id" not in df.columns:
        return [{"row": "-", "problem": "Column 'student_id' is missing — please use the official template"}], [], None
    tcols = {}
    for t in topics:
        c = next((c for c in df.columns if c.split(" (")[0] == t), None)
        if c is None:
            errors.append({"row": "-", "problem": f"Column for topic '{t}' is missing"})
        else:
            tcols[t] = c
    if errors:
        return errors, warns, None
    known = set(students.student_id)
    df["student_id"] = df.student_id.astype(str).str.strip().str.upper()
    for i, r in df.iterrows():
        row = i + 2  # Excel row number (header is row 1)
        sid = r.student_id
        if sid in ("", "NAN", "NONE"):
            errors.append({"row": row, "problem": "Student ID is empty"}); continue
        if sid not in known:
            errors.append({"row": row, "problem": f"Unknown student ID '{sid}'"})
        for t, c in tcols.items():
            v = r[c]
            if pd.isna(v) or str(v).strip() == "":
                warns.append({"row": row, "problem": f"{sid}: {t} is blank — saved as ABSENT (not counted in averages)"}); continue
            if str(v).strip().upper() in ("AB", "ABSENT", "A"):
                warns.append({"row": row, "problem": f"{sid}: {t} marked absent (not counted in averages)"}); continue
            try:
                fv = float(v)
            except ValueError:
                errors.append({"row": row, "problem": f"{sid}: {t} has '{v}' — not a number"}); continue
            if fv < 0 or fv > max_marks:
                errors.append({"row": row, "problem": f"{sid}: {t} = {fv:g} is outside 0–{max_marks}"})
    dup = df.student_id[df.student_id.duplicated()].unique()
    for d in dup:
        errors.append({"row": "-", "problem": f"Student {d} appears more than once"})
    if errors:
        return errors, warns, None
    long = df.melt(id_vars=["student_id"], value_vars=list(tcols.values()), var_name="topic", value_name="marks")
    long["topic"] = long.topic.str.split(" \\(").str[0]
    raw = long.marks.astype(str).str.strip().str.upper()
    long["absent"] = (long.marks.isna() | raw.isin(["AB", "ABSENT", "A", "", "NAN", "NONE"])).astype(int)
    long["marks"] = pd.to_numeric(long.marks.where(long.absent == 0), errors="coerce").fillna(0)
    return errors, warns, long
