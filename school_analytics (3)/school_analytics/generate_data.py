"""Generate a realistic sample dataset for the School Analytics app.
Replace the CSVs in data/ with your real data (same columns) to use real marks."""
import numpy as np, pandas as pd, os

rng = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(__file__), "data"); os.makedirs(OUT, exist_ok=True)

N, SECTIONS = 600, list("ABCD")
EXAMS = ["Unit Test 1", "Mid Term", "Unit Test 2", "Final"]
TOPICS = {
    "Mathematics": ["Algebra", "Trigonometry", "Calculus", "Probability"],
    "Physics": ["Mechanics", "Electricity", "Optics", "Modern Physics"],
    "Chemistry": ["Physical", "Organic", "Inorganic", "Practical"],
    "English": ["Grammar", "Reading", "Writing", "Literature"],
    "Computer Science": ["Programming", "Data Structures", "Databases", "Networks"],
    "Biology": ["Botany", "Zoology", "Genetics", "Human Physiology"],
}
SUBJECTS = list(TOPICS)
FIRST = ["Ravi","Lakshmi","Suresh","Anitha","Kiran","Padma","Venkat","Swathi","Mahesh","Divya","Ramesh","Sravani","Arjun","Keerthi","Naveen","Bhavana","Sai","Harika","Teja","Pooja"]
LAST = ["Reddy","Naidu","Rao","Sharma","Kumar","Varma","Chowdary","Goud","Prasad","Yadav"]

# ---- teachers: one per subject per section (some teach two sections)
teachers, tmap, tid = [], {}, 1
teacher_effect = {}
for s in SUBJECTS:
    pool = []
    for k in range(3):
        t = f"T{tid:02d}"; tid += 1
        name = f"Prof. {rng.choice(FIRST)} {rng.choice(LAST)}"
        exp = int(rng.integers(2, 25))
        teachers.append({"teacher_id": t, "teacher_name": name, "subject": s, "experience_yrs": exp})
        pool.append(t); teacher_effect[t] = rng.normal(0, 2)
    for i, sec in enumerate(SECTIONS):
        tmap[(s, sec)] = pool[[0, 1, 2, 0][i]]
# planted issue: Section C maths teacher is weak (guest faculty)
weak_t = f"T{tid:02d}"; tid += 1
teachers.append({"teacher_id": weak_t, "teacher_name": "Prof. Guest Faculty (Maths)", "subject": "Mathematics", "experience_yrs": 1})
tmap[("Mathematics", "C")] = weak_t; teacher_effect[weak_t] = -9
teachers = pd.DataFrame(teachers)

# ---- students
sec = rng.choice(SECTIONS, N)
st = pd.DataFrame({
    "student_id": [f"S{1001+i}" for i in range(N)],
    "student_name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(N)],
    "section": sec,
    "gender": rng.choice(["M", "F"], N),
    "parent_education": rng.choice(["None", "School", "Graduate", "PostGrad"], N, p=[.15, .4, .32, .13]),
    "private_tuition": rng.choice([0, 1], N, p=[.55, .45]),
    "home_internet": rng.choice([0, 1], N, p=[.3, .7]),
    "travel_time_min": np.clip(rng.gamma(2, 6, N), 1, 45).round().astype(int),
    "study_hours_per_day": np.clip(rng.normal(2.5, 1.1, N), 0, 6).round(1),
    "sleep_hours": np.clip(rng.normal(7, 1.1, N), 4, 9.5).round(1),
    "screen_time_hours": np.clip(rng.normal(3.5, 1.5, N), .5, 9).round(1),
})
ability = rng.normal(0, 1, N)
st["previous_year_pct"] = np.clip(62 + 11 * ability + rng.normal(0, 5, N), 25, 99).round(1)
pe = st.parent_education.map({"None": -4, "School": 0, "Graduate": 3, "PostGrad": 5}).values
motivation = rng.normal(0, 1, N)

# ---- monthly attendance (Jun..Feb); some students decline over the year
months = ["Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb"]
base_att = np.clip(84 + 6 * motivation + rng.normal(0, 5, N), 45, 100)
decline = (rng.random(N) < .12) * rng.uniform(1.5, 4, N)
att_rows = []
for m_i, m in enumerate(months):
    a = np.clip(base_att - decline * m_i + rng.normal(0, 4, N), 20, 100)
    att_rows.append(pd.DataFrame({"student_id": st.student_id, "month": m, "attendance_pct": a.round(1)}))
att = pd.concat(att_rows)
st["attendance_pct"] = att.groupby("student_id", sort=False).attendance_pct.mean().reindex(st.student_id).round(1).values
st["assignment_submission_pct"] = np.clip(75 + 12 * motivation + rng.normal(0, 10, N), 10, 100).round(0)
st["lab_attendance_pct"] = np.clip(st.attendance_pct + rng.normal(-8, 10, N), 15, 100).round(0)

# topic difficulty (planted: Calculus, Electricity, Organic are hard school-wide)
diff = {t: 0 for s in TOPICS for t in TOPICS[s]}
diff.update({"Calculus": -5, "Electricity": -4, "Organic": -3, "Probability": -2, "Genetics": -1.5})
subj_base = {"Mathematics": 55, "Physics": 57, "Chemistry": 62, "English": 69, "Computer Science": 67, "Biology": 64}

# planted intervention: 40 weak Maths students got remedial classes after the Mid Term
weak_idx = np.argsort(7 * ability + .35 * (base_att - 82) + rng.normal(0, 3, N))[:80]
remedial = rng.choice(weak_idx, 40, replace=False)
remedial_mask = np.zeros(N, bool); remedial_mask[remedial] = True

rows = []
for e_i, ex in enumerate(EXAMS):
    att_ex = att[att.month.isin(months[e_i * 2: e_i * 2 + 3])].groupby("student_id", sort=False).attendance_pct.mean().reindex(st.student_id).values
    for s in SUBJECTS:
        t_ids = np.array([tmap[(s, x)] for x in st.section])
        te = np.array([teacher_effect[t] for t in t_ids])
        trend = np.where(t_ids == weak_t, -1.5 * e_i, 0)  # weak teacher's class keeps sliding
        score = (subj_base[s] + 7 * ability + .35 * (att_ex - 82) + 2.5 * (st.study_hours_per_day - 2.5)
                 + .12 * (st.assignment_submission_pct - 75) - 1.2 * (st.screen_time_hours - 3.5)
                 + pe * .7 + te + trend - .12 * (st.travel_time_min - 12)).to_numpy(dtype=float, copy=True)
        if e_i == 2: score -= 6   # planted: Unit Test 2 paper was harder for everyone
        if s == "Mathematics":
            score += 5 * st.private_tuition.values
            if e_i >= 2: score += np.where(remedial_mask, 3 + 3 * (e_i - 1), 0)
        if s in ("Physics", "Chemistry"): score += .2 * (st.lab_attendance_pct.values - 70)
        if s == "Computer Science": score += 6 * st.home_internet.values
        if s in ("Mathematics", "Physics"): score += 1.2 * (st.sleep_hours.values - 7)
        for tp in TOPICS[s]:
            m = np.clip((score + diff[tp] + rng.normal(0, 7, N)) / 4, 0, 25).round(1)  # each topic out of 25
            rows.append(pd.DataFrame({"student_id": st.student_id, "exam": ex, "exam_order": e_i + 1, "subject": s,
                                      "topic": tp, "teacher_id": t_ids, "marks": m, "max_marks": 25}))
marks = pd.concat(rows, ignore_index=True)
# planted absences: ~1.2% of student-exam-subject papers missed entirely (weaker attenders more likely)
marks["absent"] = 0
key = marks[["student_id", "exam_order", "subject"]].drop_duplicates()
p_abs = .012 * (100 - st.set_index("student_id").attendance_pct.reindex(key.student_id).values) / (100 - st.attendance_pct.mean())
miss = key[rng.random(len(key)) < np.clip(p_abs, 0, .08)]
m_idx = marks.set_index(["student_id", "exam_order", "subject"]).index.isin(miss.set_index(["student_id", "exam_order", "subject"]).index)
marks.loc[m_idx, ["marks", "absent"]] = [0, 1]

st.to_csv(f"{OUT}/students.csv", index=False)
teachers.to_csv(f"{OUT}/teachers.csv", index=False)
marks.to_csv(f"{OUT}/marks.csv", index=False)
att.to_csv(f"{OUT}/attendance.csv", index=False)
# ---- last year's batch (final exam) for year-on-year comparison
ly = []
for s_ in SUBJECTS:
    for sec_ in SECTIONS:
        cur = marks[(marks.subject == s_) & (marks.exam_order == 4)].groupby("student_id").marks.sum()
        shift = {"Mathematics": 3.5, "Physics": -1.0, "Chemistry": 1.5, "English": -0.5, "Computer Science": -2.0, "Biology": 0.5}[s_]
        ly.append({"subject": s_, "section": sec_, "avg_score": round(cur.mean() + shift + rng.normal(0, 1), 1),
                   "pass_rate": round(min(100, 97 + rng.normal(0, 1.5) - (3 if s_ == "Mathematics" else 0)), 1)})
pd.DataFrame(ly).to_csv(f"{OUT}/last_year.csv", index=False)

# ---- sample action already taken (so the Action Tracker can show impact)
pd.DataFrame([{
    "action_id": 1, "created": "2026-09-20", "problem": "Many students weak in Mathematics after Mid Term",
    "reason": "Low basics in Algebra & Calculus", "action": "Free remedial Maths classes, 3 days a week",
    "owner": "HOD Mathematics", "deadline": "2026-12-15", "status": "In progress", "subject": "Mathematics",
    "start_exam_order": 2, "target_students": ";".join(st.student_id.values[remedial])}]).to_csv(f"{OUT}/actions.csv", index=False)

# ---- targets set by management
import json
json.dump({"pass_rate": 95, "school_avg": 65, "distinction": 12, "attendance": 85,
           "subject_avg": {s_: 60 for s_ in SUBJECTS}}, open(f"{OUT}/targets.json", "w"), indent=2)

# ---- user accounts (demo passwords; change after first login)
import hashlib
h = lambda pw: hashlib.sha256(("school-salt:" + pw).encode()).hexdigest()
users = [{"username": "principal", "password_hash": h("principal123"), "role": "principal", "scope": "ALL", "display_name": "Principal"}]
for s_ in SUBJECTS:
    users.append({"username": "hod_" + s_.lower().replace(" ", "_"), "password_hash": h("hod123"), "role": "hod", "scope": s_, "display_name": f"HOD {s_}"})
for t in teachers.itertuples():
    users.append({"username": t.teacher_id.lower(), "password_hash": h("teacher123"), "role": "teacher", "scope": t.teacher_id, "display_name": t.teacher_name})
for x in st.itertuples():
    users.append({"username": x.student_id.lower(), "password_hash": h("parent123"), "role": "parent", "scope": x.student_id, "display_name": f"Parent of {x.student_name}"})
pd.DataFrame(users).to_csv(f"{OUT}/users.csv", index=False)

# ======================================================================
# v3 additions: daily attendance, homework, workload, history, questions
# ======================================================================
# class teacher per section + teacher workload
class_teacher = {"A": "T01", "B": "T05", "C": "T09", "D": "T13"}
pd.DataFrame([{"section": k, "class_teacher_id": v, "room": f"Room {i+101}"} for i, (k, v) in enumerate(class_teacher.items())]).to_csv(f"{OUT}/sections.csv", index=False)
teachers["periods_per_week"] = [int(x) for x in rng.integers(22, 34, len(teachers))]
teachers.loc[teachers.teacher_id == weak_t, "periods_per_week"] = 38   # overloaded guest faculty
teachers["other_duties"] = rng.choice(["None", "Exam cell", "Sports", "Admin", "None"], len(teachers))
teachers.loc[teachers.teacher_id == weak_t, "other_duties"] = "Admin + Exam cell"
teachers.to_csv(f"{OUT}/teachers.csv", index=False)

# daily attendance: school days from 2026-06-12 up to 2026-11-06 (Mon-Sat)
days = pd.bdate_range("2026-06-12", "2026-11-06", freq="C", weekmask="Mon Tue Wed Thu Fri Sat")
p_present = (base_att / 100)[:, None] - decline[:, None] * np.linspace(0, 3, len(days))[None, :] / 100
pres = rng.random((N, len(days))) < np.clip(p_present, .2, .995)
# planted: 9 students absent the last 3-6 days in a row
streak_ids = rng.choice(N, 9, replace=False)
for i in streak_ids:
    pres[i, -int(rng.integers(3, 7)):] = False
da = pd.DataFrame(pres.astype(int), columns=[d.date().isoformat() for d in days])
da.insert(0, "student_id", st.student_id.values)
da = da.melt(id_vars="student_id", var_name="date", value_name="present")
da.to_csv(f"{OUT}/daily_attendance.csv", index=False)

# homework: 2 per subject per month (Jul-Oct)
hw_rows, hw_meta = [], []
hid = 1
for s_ in SUBJECTS:
    for d in pd.date_range("2026-07-05", "2026-10-30", freq="14D"):
        hw_meta.append({"hw_id": f"H{hid:03d}", "subject": s_, "date": d.date().isoformat(), "title": f"{s_} worksheet {hid}"})
        p_sub = np.clip(st.assignment_submission_pct.values / 100, .05, .99)
        r_ = rng.random(N)
        status = np.where(r_ < p_sub, "Submitted", np.where(r_ < p_sub + .08, "Late", "Missing"))
        for sid, stt in zip(st.student_id.values, status):
            hw_rows.append((f"H{hid:03d}", sid, stt))
        hid += 1
pd.DataFrame(hw_meta).to_csv(f"{OUT}/homework.csv", index=False)
pd.DataFrame(hw_rows, columns=["hw_id", "student_id", "status"]).to_csv(f"{OUT}/homework_status.csv", index=False)

# multi-year history: same students in Class 8 and 9 (this year = Class 10)
hist = []
for yr, cls, drift in [(2024, 8, 2.0), (2025, 9, 1.0)]:
    for s_ in SUBJECTS:
        sc = np.clip(subj_base[s_] + drift + 7 * ability + rng.normal(0, 7, N) + (3 if s_ == "Mathematics" else 0), 5, 100).round(0)
        hist += [{"student_id": a_, "year": yr, "class": cls, "subject": s_, "final_score": b_} for a_, b_ in zip(st.student_id.values, sc)]
pd.DataFrame(hist).to_csv(f"{OUT}/history.csv", index=False)

# question-level marks for the Final: each topic = 5 questions x 5 marks
# question types: Recall (easy) / Application / Problem-solving (hard)
qtypes = ["Recall", "Recall", "Application", "Application", "Problem-solving"]
qdiff = {"Recall": 1.0, "Application": 0.0, "Problem-solving": -1.2}
careless = rng.random(N) < .15   # some students lose marks on easy questions
fin = marks[marks.exam_order == 4]
qrows, qmeta = [], []
for (s_, tp), g in fin.groupby(["subject", "topic"]):
    g = g.set_index("student_id").reindex(st.student_id)
    lvl = (g.marks.values / 25 - .5) * 4  # ability on this topic
    for qi, qt in enumerate(qtypes, 1):
        qid = f"{s_[:3].upper()}-{tp[:4].upper()}-Q{qi}"
        qmeta.append({"question_id": qid, "subject": s_, "topic": tp, "q_no": qi, "type": qt, "max_marks": 5})
        pr = 1 / (1 + np.exp(-(lvl + qdiff[qt] * 1.6)))
        if qt == "Recall": pr = np.where(careless, pr * .55, pr)
        m_ = np.clip(np.round(pr * 5 + rng.normal(0, .7, N)), 0, 5)
        qrows += list(zip(st.student_id.values, [qid] * N, m_))
pd.DataFrame(qmeta).to_csv(f"{OUT}/questions.csv", index=False)
pd.DataFrame(qrows, columns=["student_id", "question_id", "marks"]).to_csv(f"{OUT}/question_marks.csv", index=False)

print("students", st.shape, "| marks", marks.shape, "| teachers", teachers.shape, "| attendance", att.shape)
