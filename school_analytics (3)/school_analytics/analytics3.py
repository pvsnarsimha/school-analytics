"""Stronger, more honest methods — each written so the result can be explained in plain words.

1. explain_risk      — why THIS student is at risk (SHAP)
2. fairness_check    — is the warning system equally accurate for every group?
3. matched_impact    — did an action work, compared with SIMILAR students who did not get it?
4. fair_teacher_effects / fair_class_effects — multilevel model: fair comparison after accounting for students' past marks
"""
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import analytics as A

warnings.filterwarnings("ignore")


# ====================================================================== 1. explain each student's risk
def train_risk(ss, students, attendance):
    last = ss.exam_order.max()
    X = A.build_features(ss, students, attendance, last - 1)
    fin = ss[ss.exam_order == last].groupby("student_id").score.agg(["min", "mean"])
    y = ((fin["min"] < A.PASS) | (fin["mean"] < 45)).astype(int).reindex(X.index).fillna(0)
    m = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0).fit(X, y)
    return m, X, y


def _shap_values(model, X):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return pd.DataFrame(sv, index=X.index, columns=X.columns)


PLAIN = {  # feature -> (label, how to show value)
    "last_exam_avg": ("Average in last exam", "{:.0f}"), "past_avg": ("Average so far this year", "{:.0f}"),
    "score_trend": ("Change in marks since first exam", "{:+.0f}"), "min_subject_last": ("Weakest subject in last exam", "{:.0f}"),
    "subjects_below_40": ("Subjects below 40 in last exam", "{:.0f}"), "attendance_trend": ("Attendance change per month", "{:+.1f}%"),
    "attendance_pct": ("Attendance", "{:.0f}%"), "assignment_submission_pct": ("Homework submitted", "{:.0f}%"),
    "study_hours_per_day": ("Study hours per day", "{:.1f}"), "sleep_hours": ("Sleep hours", "{:.1f}"),
    "screen_time_hours": ("Screen time (hours)", "{:.1f}"), "travel_time_min": ("Travel time (min)", "{:.0f}"),
    "lab_attendance_pct": ("Lab attendance", "{:.0f}%"), "private_tuition": ("Private tuition", "{}"),
    "home_internet": ("Internet at home", "{}"), "previous_year_pct": ("Last year's result", "{:.0f}%"),
}
ACTIONABLE = {"attendance_pct", "attendance_trend", "assignment_submission_pct", "study_hours_per_day", "screen_time_hours", "sleep_hours", "lab_attendance_pct"}


def explain_risk(ss, students, attendance, top=3):
    """For every student: risk %, and the top reasons pushing risk UP and DOWN, in plain sentences."""
    m, X, y = train_risk(ss, students, attendance)
    prob = pd.Series(m.predict_proba(X)[:, 1], index=X.index)
    sv = _shap_values(m, X) * 100  # in percentage points of risk
    rows = []
    for sid in X.index:
        s = sv.loc[sid].sort_values(ascending=False)
        def fmt(f):
            lab, f_ = PLAIN.get(f, (f, "{}"))
            v = X.loc[sid, f]
            if f in ("private_tuition", "home_internet"): v = "Yes" if v else "No"
            return f"{lab}: {f_.format(v)}"
        up = [(fmt(f), v, f in ACTIONABLE) for f, v in s.items() if v > 1][:top]
        down = [(fmt(f), v) for f, v in s[::-1].items() if v < -1][:2]
        rows.append({"student_id": sid, "risk": prob[sid] * 100,
                     "raises_risk": "; ".join(f"{t} (+{v:.0f})" for t, v, _ in up),
                     "lowers_risk": "; ".join(f"{t} ({v:.0f})" for t, v in down),
                     "can_change": "; ".join(t.split(":")[0] for t, v, a in up if a) or "—"})
    return pd.DataFrame(rows).set_index("student_id"), sv, X


def student_story(sid, expl, sv, X, name=None):
    """A short paragraph a class teacher can read out at a parent meeting."""
    r = expl.loc[sid]; s = sv.loc[sid].sort_values(ascending=False)
    who = name.split()[0] if name else sid
    parts = []
    for f, v in s.items():
        if v <= 1 or len(parts) >= 3: break
        lab, f_ = PLAIN.get(f, (f, "{}")); val = X.loc[sid, f]
        if f in ("private_tuition", "home_internet"): val = "Yes" if val else "No"
        parts.append(f"{lab.lower()} is {f_.format(val)} (adds about {v:.0f} points)")
    if not parts:
        return f"{who}'s risk is low ({r.risk:.0f}%). Nothing stands out — keep encouraging."
    act = r.can_change if r.can_change != "—" else "marks in weak subjects (extra classes)"
    return (f"{who} has a {r.risk:.0f}% chance of failing at least one subject. The main reasons: " + "; ".join(parts) +
            f". What can change quickly: {act.lower()}.")


# ====================================================================== 2. fairness
def fairness_check(ss, students, attendance):
    """Time-based honest test, then compare accuracy across groups of students."""
    last = ss.exam_order.max()
    def xy(upto, target):
        X = A.build_features(ss, students, attendance, upto)
        f = ss[ss.exam_order == target].groupby("student_id").score.agg(["min", "mean"])
        return X, ((f["min"] < A.PASS) | (f["mean"] < 45)).astype(int).reindex(X.index).fillna(0)
    Xtr, ytr = xy(last - 2, last - 1); Xte, yte = xy(last - 1, last)
    m = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    pred = pd.Series(m.predict_proba(Xte)[:, 1] >= .5, index=Xte.index)
    st_ = students.set_index("student_id").reindex(Xte.index)
    groups = {"Gender": st_.gender.map({"M": "Boys", "F": "Girls"}), "Internet at home": st_.home_internet.map({1: "Has internet", 0: "No internet"}),
              "Section": "Section " + st_.section, "Parents' education": st_.parent_education, "Private tuition": st_.private_tuition.map({1: "Tuition", 0: "No tuition"})}
    rows = []
    for gname, g in groups.items():
        for val in sorted(g.dropna().unique()):
            mask = g == val
            yt, pr = yte[mask], pred[mask]
            fails, passes = (yt == 1).sum(), (yt == 0).sum()
            rows.append({"group": gname, "value": val, "students": int(mask.sum()), "actually_failed_pct": yt.mean() * 100, "flagged_pct": pr.mean() * 100,
                         "caught_pct": (pr & (yt == 1)).sum() / fails * 100 if fails else np.nan,
                         "false_alarm_pct": (pr & (yt == 0)).sum() / passes * 100 if passes else np.nan, "fails": int(fails)})
    r = pd.DataFrame(rows)
    verdicts = []
    for gname, g in r.groupby("group"):
        c, f = g.caught_pct, g.false_alarm_pct
        gap_c, gap_f = c.max() - c.min(), f.max() - f.min()
        small = g.fails.min() < 8
        if gap_c > 20 and not small:
            v = ("🔴", f"Misses more failing students in '{g.loc[c.idxmin(), 'value']}' ({c.min():.0f}% caught vs {c.max():.0f}%). Teachers should double-check this group manually.")
        elif gap_f > 10:
            v = ("🟡", f"More false alarms for '{g.loc[f.idxmax(), 'value']}' ({f.max():.0f}% vs {f.min():.0f}%). Do not treat a warning as a label.")
        elif small:
            v = ("🟡", f"Too few failing students in some groups to judge fairly (smallest group has {g.fails.min()}). Check again with more data.")
        else:
            v = ("🟢", f"Works about equally well for all {gname.lower()} groups (caught {c.min():.0f}–{c.max():.0f}%).")
        verdicts.append({"group": gname, "status": v[0], "verdict": v[1]})
    return r, pd.DataFrame(verdicts)


# ====================================================================== 3. matched comparison for actions
def matched_impact(ss, students, action, n_boot=1000, seed=0):
    """Compare targeted students with the most SIMILAR students who did not get the action
    (same subject marks at the start, attendance and last-year result). Removes the 'weak students bounce back anyway' illusion."""
    ids = [x for x in str(action.get("target_students", "")).split(";") if x]
    subj, start = action.get("subject"), int(action.get("start_exam_order", 0) or 0)
    last = ss.exam_order.max()
    if not ids or not subj or start == 0 or start >= last:
        return None
    s = ss[ss.subject == subj].pivot_table(index="student_id", columns="exam_order", values="score")[[start, last]].dropna()
    s.columns = ["before", "after"]
    d = s.join(students.set_index("student_id")[["attendance_pct", "previous_year_pct"]])
    d["change"] = d.after - d.before
    T, C = d[d.index.isin(ids)], d[~d.index.isin(ids)]
    if len(T) < 5 or len(C) < len(T):
        return None
    feats = ["before", "attendance_pct", "previous_year_pct"]; w = np.array([2.0, 1.0, 1.0])  # starting marks matter most
    mu, sd = d[feats].mean(), d[feats].std()
    Tz, Cz = ((T[feats] - mu) / sd).values * w, ((C[feats] - mu) / sd).values * w
    used, pairs = set(), []
    for i in np.argsort(-np.abs(Tz[:, 0])):  # match hardest cases first
        dist = np.sqrt(((Cz - Tz[i]) ** 2).sum(1)); dist[list(used)] = np.inf
        j = int(np.argmin(dist)); used.add(j); pairs.append((i, j, dist[j]))
    ti, ci, dist = map(np.array, zip(*pairs))
    tch, cch = T.change.values[ti], C.change.values[ci]
    eff = tch - cch
    rng = np.random.default_rng(seed)
    boots = [rng.choice(eff, len(eff)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    naive = T.change.mean() - C.change.mean()
    balance = pd.DataFrame({"Targeted students": T[feats].values[ti].mean(0), "Matched similar students": C[feats].values[ci].mean(0),
                            "All other students": C[feats].mean().values}, index=["Marks at start", "Attendance %", "Last year %"]).round(1)
    worked = lo > 0
    verdict = ("✅ It worked" if worked and eff.mean() >= 2 else "🟡 Small effect" if worked else "⏳ No clear effect yet" if hi > 0 else "🔴 Did not help")
    return {"n": len(eff), "target_change": tch.mean(), "matched_change": cch.mean(), "others_change": C.change.mean(),
            "effect": eff.mean(), "lo": lo, "hi": hi, "naive": naive, "verdict": verdict, "worked": worked, "balance": balance,
            "before": T.before.mean(), "after": T.after.mean(), "match_quality": "Good" if np.median(dist) < .5 else "Fair" if np.median(dist) < 1 else "Weak",
            "sentence": (f"Similar students who did NOT get this action changed by {cch.mean():+.1f} marks on their own. "
                         f"The {len(eff)} targeted students changed by {tch.mean():+.1f}. So the action itself added about "
                         f"{eff.mean():+.1f} marks (likely between {lo:+.1f} and {hi:+.1f}).")}


# ====================================================================== 4. multilevel model
def _mixed(df, group_col):
    """Random-intercept (multilevel) estimate by empirical Bayes:
    1) predict each student's score from subject + last-year result + attendance + homework (ignoring who teaches them),
    2) average each group's leftover (actual - predicted),
    3) shrink each group toward 0 by how reliable it is: small groups are pulled in more.
    Between-group variance is estimated by the method of moments."""
    df = df.dropna(subset=["score", "previous_year_pct", "attendance_pct", "assignment_submission_pct"]).copy()
    Xd = pd.get_dummies(df.subject, drop_first=True, dtype=float)
    Xm = np.column_stack([np.ones(len(df)), Xd.values, df[["previous_year_pct", "attendance_pct", "assignment_submission_pct"]].values])
    beta, *_ = np.linalg.lstsq(Xm, df.score.values, rcond=None)
    df["resid"] = df.score.values - Xm @ beta
    g = df.groupby(group_col).resid
    mean_r, n = g.mean(), g.size()
    sigma2 = df.groupby(group_col).resid.var(ddof=1).mean()           # within-group (student) variance
    tau2 = max(mean_r.var(ddof=1) - (sigma2 / n).mean(), 1e-6)       # between-group variance
    w = tau2 / (tau2 + sigma2 / n)                                     # reliability of each group's mean
    eff = w * mean_r
    post_sd = np.sqrt(w * sigma2 / n)
    out = pd.DataFrame({"group": mean_r.index, "effect": eff.values, "lo": (eff - 1.96 * post_sd).values, "hi": (eff + 1.96 * post_sd).values,
                        "students": n.values, "unshrunk": mean_r.values, "reliability": (w * 100).round(0).values})
    out["shrink_pct"] = (100 - out.reliability)
    out["verdict"] = np.select([(out.hi < 0) & (out.effect <= -3), (out.hi < 0) & (out.effect <= -1.5), (out.lo > 0) & (out.effect >= 3), (out.lo > 0) & (out.effect >= 1.5)],
                               ["Below expected", "Slightly below", "Above expected", "Slightly above"], "As expected")
    r2 = 1 - df.resid.var() / df.score.var()
    return out, {"between_group_sd": np.sqrt(tau2), "student_sd": np.sqrt(sigma2), "icc": tau2 / (tau2 + sigma2), "r2_background": r2}


def fair_teacher_effects(ss, students, teachers, exam):
    """Each teacher's effect on marks after accounting for subject, students' last-year result, attendance and homework.
    Small classes are pulled toward the average so one unlucky group does not unfairly label a teacher."""
    df = ss[ss.exam == exam].merge(students[["student_id", "previous_year_pct", "attendance_pct", "assignment_submission_pct"]], on="student_id")
    out, info = _mixed(df, "teacher_id")
    out = out.rename(columns={"group": "teacher_id"}).merge(teachers, on="teacher_id")
    raw = df.groupby("teacher_id").score.mean() - df.groupby("subject").score.transform("mean").groupby(df.teacher_id).mean()
    out["raw_gap"] = out.teacher_id.map(raw)
    return out.sort_values("effect"), info


def fair_class_effects(ss, students, exam):
    """Same, for each class (subject × section)."""
    df = ss[ss.exam == exam].merge(students[["student_id", "section", "previous_year_pct", "attendance_pct", "assignment_submission_pct"]], on="student_id")
    df["klass"] = df.subject + " · Section " + df.section
    out, info = _mixed(df, "klass")
    return out.rename(columns={"group": "class"}).sort_values("effect"), info


def explain_effect(r, who):
    rng_ = f"likely between {r.lo:+.1f} and {r.hi:+.1f}"
    if r.verdict in ("Below expected", "Slightly below"):
        return f"After allowing for students' past marks, attendance and homework, {who}'s students score about **{abs(r.effect):.1f} marks less** than expected ({rng_})."
    if r.verdict in ("Above expected", "Slightly above"):
        return f"After allowing for students' past marks, attendance and homework, {who}'s students score about **{r.effect:.1f} marks more** than expected ({rng_})."
    return f"{who}'s students score as expected for students like them ({rng_}) — no real difference."
