"""v3 analysis: attendance early warning, homework, targets, workload, multi-year, mistakes, honest model check, pilot tools."""
import hashlib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import analytics as A


# ---------------------------------------------------------------- 1. early warning
def absence_streaks(daily, upto=None):
    """Consecutive absent days ending on the latest school day, per student."""
    d = daily if upto is None else daily[daily.date <= upto]
    days = sorted(d.date.unique())
    wide = d.pivot_table(index="student_id", columns="date", values="present", aggfunc="max")[days]
    absent = (wide.values == 0)
    # count trailing absences
    rev = absent[:, ::-1]
    streak = np.where(rev.all(1), rev.shape[1], np.argmin(rev, axis=1))
    last30 = wide.iloc[:, -30:].mean(1) * 100
    return pd.DataFrame({"student_id": wide.index, "absent_streak": streak, "attendance_last_30_days": last30.round(0).values,
                         "last_present": [days[len(days) - 1 - s] if s < len(days) else "—" for s in streak]})


def early_warnings(daily, students, sections, hw_student=None, min_streak=3):
    s = absence_streaks(daily).merge(students[["student_id", "student_name", "section"]], on="student_id")
    s = s.merge(sections[["section", "class_teacher_id"]], on="section", how="left")
    out = []
    for r in s[s.absent_streak >= min_streak].itertuples():
        out.append({"student_id": r.student_id, "student_name": r.student_name, "section": r.section, "class_teacher_id": r.class_teacher_id,
                    "alert": f"Absent {r.absent_streak} school days in a row", "level": "High" if r.absent_streak >= 5 else "Medium",
                    "action": "Call parent today"})
    for r in s[(s.absent_streak < min_streak) & (s.attendance_last_30_days < 65)].itertuples():
        out.append({"student_id": r.student_id, "student_name": r.student_name, "section": r.section, "class_teacher_id": r.class_teacher_id,
                    "alert": f"Only {r.attendance_last_30_days:.0f}% attendance in last 30 days", "level": "Medium", "action": "Talk to student; inform parent"})
    if hw_student is not None:
        bad = hw_student[hw_student.missing_last3 >= 3].merge(students[["student_id", "student_name", "section"]], on="student_id")
        bad = bad.merge(sections[["section", "class_teacher_id"]], on="section", how="left")
        for r in bad.itertuples():
            out.append({"student_id": r.student_id, "student_name": r.student_name, "section": r.section, "class_teacher_id": r.class_teacher_id,
                        "alert": f"Missed last 3 {r.subject} homeworks", "level": "Medium", "action": f"{r.subject} teacher to check"})
    return pd.DataFrame(out, columns=["student_id", "student_name", "section", "class_teacher_id", "alert", "level", "action"])


# ---------------------------------------------------------------- 2. homework
def homework_tables(hw, hws):
    j = hws.merge(hw, on="hw_id")
    j["done"] = j.status.isin(["Submitted", "Late"]).astype(int)
    per = j.groupby(["student_id", "subject"]).agg(submitted_pct=("done", lambda x: x.mean() * 100),
                                                   missing=("status", lambda x: (x == "Missing").sum())).reset_index()
    j = j.sort_values("date")
    last3 = j.groupby(["student_id", "subject"]).tail(3).groupby(["student_id", "subject"]).status.apply(lambda x: (x == "Missing").sum()).rename("missing_last3")
    per = per.merge(last3.reset_index(), on=["student_id", "subject"])
    return j, per


def homework_effect(per, ss, exam):
    e = ss[ss.exam == exam][["student_id", "subject", "score"]].merge(per, on=["student_id", "subject"])
    e["band"] = pd.cut(e.submitted_pct, [-1, 50, 75, 90, 101], labels=["Below 50%", "50–75%", "75–90%", "90%+"])
    return e.groupby("band", observed=True).score.agg(["mean", "size"]).reset_index(), float(np.corrcoef(e.submitted_pct, e.score)[0, 1])


# ---------------------------------------------------------------- 3. per-student targets
def student_targets(ss, history, sid=None):
    """Realistic next-exam target per student & subject from their own trend (this year + previous years)."""
    piv = ss.pivot_table(index=["student_id", "subject"], columns="exam_order", values="score")
    if sid is not None:
        piv = piv.loc[[sid]]
    hist = history.pivot_table(index=["student_id", "subject"], columns="year", values="final_score")
    x = np.array(piv.columns, float)
    # a missed (absent) exam is filled from the student's neighbouring exams — for the trend only
    piv = piv.T.interpolate(limit_direction="both").T
    Y = piv.values
    slope = ((x - x.mean()) * (Y - Y.mean(1, keepdims=True))).sum(1) / ((x - x.mean()) ** 2).sum()
    last = Y[:, -1]
    best = np.nanmax(np.column_stack([Y, hist.reindex(piv.index).values]), axis=1)
    projected = last + np.clip(slope, -3, 3)
    gap_to_best = np.clip(best - last, 0, 15)
    stretch = 3 + 0.4 * gap_to_best          # aim to win back part of the gap to the student's own best
    target = np.clip(np.round(np.maximum(projected, last) + stretch), A.PASS + 5, 98)
    out = piv.reset_index()[["student_id", "subject"]]
    out["current"] = last.round(0); out["best_ever"] = best.round(0); out["trend_per_exam"] = slope.round(1); out["target_next_exam"] = target
    out["sentence"] = [f"should reach {int(t)} in {s} next exam (now {int(c)}, best ever {int(b)})" for s, t, c, b in zip(out.subject, out.target_next_exam, out.current, out.best_ever)]
    return out


# ---------------------------------------------------------------- 4. workload
def teacher_workload(teachers, ss, students, exam):
    va = A.value_added(ss, students, teachers, exam)[["teacher_id", "value_added", "avg_score", "students"]]
    e = ss[ss.exam == exam].merge(students[["student_id", "section"]], on="student_id")
    sec = e.groupby("teacher_id").section.agg(lambda x: ", ".join(sorted(x.unique()))).rename("sections")
    ncls = e.groupby("teacher_id").section.nunique().rename("classes")
    w = teachers.merge(va, on="teacher_id").merge(sec, on="teacher_id").merge(ncls, on="teacher_id")
    w["avg_class_size"] = (w.students / w.classes).round(0)
    w["load"] = np.select([w.periods_per_week >= 34, w.periods_per_week >= 30], ["Overloaded", "Heavy"], "Normal")
    return w


# ---------------------------------------------------------------- 5. multi-year
def multi_year(history, ss, sid=None):
    fin = ss[ss.exam_order == ss.exam_order.max()][["student_id", "subject", "score"]].rename(columns={"score": "final_score"})
    cur_year = int(history.year.max()) + 1
    fin = fin.assign(year=cur_year, **{"class": int(history["class"].max()) + 1})
    allh = pd.concat([history, fin], ignore_index=True)
    if sid is not None:
        allh = allh[allh.student_id == sid]
    return allh


def cohort_movement(allh):
    """How many students improved / dropped between the last two years per subject."""
    yrs = sorted(allh.year.unique())[-2:]
    p = allh[allh.year.isin(yrs)].pivot_table(index=["student_id", "subject"], columns="year", values="final_score")
    d = (p[yrs[1]] - p[yrs[0]]).rename("change").reset_index()
    d["movement"] = np.select([d.change >= 5, d.change <= -5], ["Improved 5+", "Dropped 5+"], "Stable")
    return d.groupby(["subject", "movement"]).size().unstack(fill_value=0).reset_index()


# ---------------------------------------------------------------- 6. mistake analysis
def mistake_profile(qm, qs, sid=None):
    j = qm.merge(qs, on="question_id")
    if sid is not None:
        j = j[j.student_id == sid]
    t = j.groupby(["student_id", "subject", "type"]).apply(lambda g: g.marks.sum() / g.max_marks.sum() * 100, include_groups=False).unstack()
    t = t.reindex(columns=["Recall", "Application", "Problem-solving"])
    def label(r):
        if r.Recall < r.Application - 12: return "Careless mistakes"
        if r.Recall < 50 and r.Application < 45: return "Concept gap"
        if r["Problem-solving"] < r.Application - 30 and r.Application >= 60: return "Needs hard-problem practice"
        if r.Recall >= 75 and r.Application >= 60: return "Strong"
        return "Steady — practise more"
    t["diagnosis"] = t.apply(label, axis=1)
    return t.round(0).reset_index()


MISTAKE_ACTION = {"Careless mistakes": "Knows the concept but loses easy marks — teach checking habits, timed practice, re-reading questions.",
                  "Concept gap": "Basics are missing — re-teach the concept from the start with simple examples.",
                  "Needs hard-problem practice": "Understands basics — give multi-step problems and past board questions.",
                  "Strong": "Give enrichment / olympiad-level questions.",
                  "Steady — practise more": "Regular practice worksheets on application questions."}


def question_stats(qm, qs):
    j = qm.merge(qs, on="question_id")
    g = j.groupby(["question_id", "subject", "topic", "q_no", "type"]).marks.mean().div(5).mul(100).rename("avg_pct").reset_index()
    return g


# ---------------------------------------------------------------- 7. honest model check
def evaluate_model_honestly(ss, students, attendance):
    """Time-based check: train on (Unit Test 1 → Mid Term) to predict (Unit Test 2 → Final), as it would be used in real life."""
    last = ss.exam_order.max()
    def xy(upto, target_exam):
        X = A.build_features(ss, students, attendance, upto)
        f = ss[ss.exam_order == target_exam].groupby("student_id").score.agg(["min", "mean"])
        y = ((f["min"] < A.PASS) | (f["mean"] < 45)).astype(int).reindex(X.index).fillna(0)
        return X, y
    if last < 3:
        return None
    Xtr, ytr = xy(last - 2, last - 1)
    Xte, yte = xy(last - 1, last)
    m = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    pred = p >= .5
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    calib = pd.DataFrame({"p": p, "y": yte.values})
    calib["bin"] = pd.cut(calib.p, [0, .2, .4, .6, .8, 1.0], include_lowest=True, labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
    cal = calib.groupby("bin", observed=True).agg(students=("y", "size"), actually_failed=("y", lambda x: x.mean() * 100)).reset_index()
    # simple baseline: "flag everyone who failed something in the last exam"
    lastfail = (ss[ss.exam_order == last - 1].groupby("student_id").score.min() < A.PASS).reindex(Xte.index).fillna(False)
    b_tp = int((lastfail & (yte == 1)).sum())
    return {"auc": roc_auc_score(yte, p) if yte.nunique() > 1 else np.nan, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "actual": int(yte.sum()), "flagged": int(pred.sum()), "calibration": cal, "baseline_caught": b_tp,
            "baseline_flagged": int(lastfail.sum())}


# ---------------------------------------------------------------- 8. pilot tools
def anonymize(df, id_col, name_cols, prefix="S", salt="pilot"):
    """Replace real IDs with stable anonymous IDs and drop name columns. Returns (anon_df, key_df)."""
    df = df.copy()
    ids = df[id_col].astype(str)
    uniq = sorted(ids.unique(), key=lambda x: hashlib.sha256((salt + x).encode()).hexdigest())
    mp = {u: f"{prefix}{1001 + i}" for i, u in enumerate(uniq)}
    key = pd.DataFrame({"original_id": list(mp), "anonymous_id": list(mp.values())})
    if name_cols:
        names = df[[id_col] + name_cols].drop_duplicates(id_col).astype(str)
        key = key.merge(names.rename(columns={id_col: "original_id"}), on="original_id", how="left")
    df[id_col] = ids.map(mp)
    df = df.drop(columns=[c for c in name_cols if c in df.columns])
    return df, key


IMPORT_FIELDS = {"student_id": "Student ID / Roll no / Admission no", "student_name": "Student name (optional)",
                 "section": "Section / Class (optional)", "subject": "Subject (if marks are in rows)", "marks": "Marks (if marks are in rows)"}


def wide_to_long(df, id_col, subject_cols, max_marks):
    """Common school format: one row per student, one column per subject → app's per-subject totals (out of 100)."""
    long = df[[id_col] + subject_cols].melt(id_vars=id_col, var_name="subject", value_name="marks")
    long["marks"] = pd.to_numeric(long.marks.replace({"AB": 0, "Ab": 0, "A": 0}), errors="coerce")
    long["score"] = (long.marks / max_marks * 100).round(1)
    return long.rename(columns={id_col: "student_id"})
