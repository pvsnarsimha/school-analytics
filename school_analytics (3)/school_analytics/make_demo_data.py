"""Create demo_data/ — a copy of data/ with realistic problems added, so every agent has something to show.
Used for the documentation screenshots.   python make_demo_data.py   then   SCHOOL_DATA_DIR=demo_data streamlit run app.py
(The real data/ folder is never touched.)"""
import os, shutil, pandas as pd

src, dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"), os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
P = lambda f: os.path.join(dst, f)

# 1. data-entry mistakes a teacher might make
m = pd.read_csv(P("marks.csv"))
def row(sid, exam, subj, topic): return m.index[(m.student_id == sid) & (m.exam == exam) & (m.subject == subj) & (m.topic == topic)][0]
m.loc[row("S1085", "Final", "Physics", "Mechanics"), "marks"] = 22.5 * 10 / 10 * 1.2      # 27 out of 25
m.loc[row("S1311", "Final", "Mathematics", "Algebra"), "marks"] = 240.0                    # 240 out of 25 (extra digit)
m.loc[row("S1402", "Mid Term", "English", "Grammar"), "marks"] = -18.0                     # minus sign typed
sel = (m.student_id == "S1500") & (m.exam == "Final") & (m.subject == "Mathematics")
m.loc[sel, "student_id"] = "S5100"                                                          # roll number mistyped (real: S1500)
# a sudden drop that may be a genuine entry mistake
m.loc[(m.student_id == "S1222") & (m.exam == "Final") & (m.subject == "Biology"), "marks"] = 3.0
m.to_csv(P("marks.csv"), index=False)

# 2. more actions at different stages so the Action Tracker agent has variety
a = pd.read_csv(P("actions.csv"), dtype={"target_students": str})
ss = m[(m.exam == "Mid Term") & (m.subject == "Physics")].groupby("student_id").apply(lambda g: g.marks.sum() / g.max_marks.sum() * 100, include_groups=False)
lows = ss[ss < 45].index.tolist()[:40]
chem = m[(m.exam == "Unit Test 1") & (m.subject == "Chemistry")].groupby("student_id").apply(lambda g: g.marks.sum() / g.max_marks.sum() * 100, include_groups=False)
extra = [
    {"action_id": 2, "created": "2026-09-20", "problem": "Physics weak after Mid Term", "reason": "Electricity & Mechanics basics low", "action": "Weekly Physics doubt-clearing hour",
     "owner": "HOD Physics", "deadline": "2026-11-10", "status": "In progress", "subject": "Physics", "start_exam_order": 2, "target_students": ";".join(lows)},
    {"action_id": 3, "created": "2026-08-30", "problem": "Chemistry marks low in Unit Test 1", "reason": "Practical work missing", "action": "Extra lab sessions every Friday",
     "owner": "HOD Chemistry", "deadline": "2026-10-30", "status": "In progress", "subject": "Chemistry", "start_exam_order": 1, "target_students": ";".join(chem[chem < 45].index[:35])},
    {"action_id": 4, "created": "2026-11-15", "problem": "Attendance below 75% in Section D", "reason": "Travel problems", "action": "Parent meeting + bus-route review",
     "owner": "Class teachers", "deadline": "2026-11-18", "status": "Not started", "subject": "", "start_exam_order": 4, "target_students": ""},
]
# the Friday lab action did NOT help: its students did a little worse than similar students in the Final
tg = [s_ for s_ in extra[1]["target_students"].split(";")]
mk = (m.exam == "Final") & (m.subject == "Chemistry") & m.student_id.isin(tg)
m.loc[mk, "marks"] = (m.loc[mk, "marks"] - 3.5).clip(lower=0)
# students who followed a remedial plan (see step 3) gain different amounts at Unit Test 2 — gives the re-check a mix of results
for sid, bump in {"S1105": 3.4, "S1331": 2.6, "S1124": 1.0}.items():
    k = (m.exam_order == 3) & (m.subject == "Physics") & (m.student_id == sid) & (m.absent == 0)
    m.loc[k, "marks"] = (m.loc[k, "marks"] + bump).clip(upper=25)
m.to_csv(P("marks.csv"), index=False)
pd.concat([a, pd.DataFrame(extra)], ignore_index=True).to_csv(P("actions.csv"), index=False)
print("demo_data/ created")

# 3. a few remedial plans approved after the Mid Term, to be re-checked at Unit Test 2 (exam 3)
import json
sc = m.assign(p=m.marks / m.max_marks * 100)
mid = sc[(sc.exam_order == 2) & (sc.subject == "Physics") & (sc.absent == 0)].groupby("student_id").p.mean().sort_values()
st_ = pd.read_csv(P("students.csv")).set_index("student_id")
diag = ["Concept gap", "Careless mistakes", "Needs hard-problem practice", "Concept gap", "Careless mistakes"]
plans = []
for sid, dg in zip(mid.index[:5], diag):
    b = round(float(mid[sid]))
    plans.append({"plan_id": f"RP-{sid}-PHY-2", "created": "2026-10-12", "student_id": sid, "student_name": st_.student_name[sid], "section": st_.section[sid], "subject": "Physics",
                  "topic": "Electricity", "diagnosis": dg, "focus": "", "baseline": b, "target": b + 8, "baseline_exam_order": 2, "recheck_exam_order": 3,
                  "owner": "T04", "owner_name": "", "days": [], "worksheets": {}, "success": "", "status": "Active"})
json.dump(plans, open(P("remedial_plans.json"), "w"), indent=1)
print("demo plans:", [(p["student_id"], p["baseline"]) for p in plans])
