# School Performance Analytics

A school-wide analytics system with separate logins for Principal, HODs, Teachers and Parents.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501 and sign in.

| Role | Username | Password |
|---|---|---|
| Principal | principal | principal123 |
| HOD | hod_mathematics, hod_physics, hod_chemistry, hod_english, hod_computer_science, hod_biology | hod123 |
| Teacher | t01 … t19 | teacher123 |
| Parent | student ID, e.g. s1085 | parent123 |

**Change these passwords (Settings page) before real use.**

## What each role sees
- **Principal**: Principal Summary (headline, traffic lights, top 3 problems with action/owner/deadline, goals, action results), School Details, Subjects & Topics, Teachers (fair value-added comparison), Students at Risk, Deep Analysis, Action Tracker, Report Cards, PDF Reports, Marks Entry, Settings (targets)
- **HOD**: only their subject — topics, question-paper check, their teachers, actions, reports, marks entry
- **Teacher**: only their own classes — weak topics, attention list, their students' report cards, marks entry
- **Parent**: only their child — report card in English or Telugu, PDF download

## Features
1. Plain language everywhere; technical details hidden behind "Show technical details"
2. One-page Principal Summary
3. Every problem written as Problem → Why → Action → Owner → Deadline
4. Action Tracker that measures whether an action worked after the next exam
5. Automatic PDFs: Principal (2 pages), HOD, Teacher sheet, Parent report card (English + Telugu), ZIP per section
6. Excel marks template with error checking (out-of-range, non-numbers, unknown IDs, duplicates, absentees)
7. Role-based access
8. Targets set by the Principal, shown as progress bars
9. "What does this mean?" on charts, max 4 charts per page, print-friendly mode, this year vs last year

## New in version 3
**Pilot support** — 🧪 Pilot Toolkit: checklist, anonymiser (removes names, creates ID key for the school), feedback form, and an honest accuracy test of the prediction on an exam it has never seen.

**Easy data entry**
- 📅 Attendance: tick absent students on phone or computer (~30 s per class)
- 📝 Marks Entry has 3 ways: Excel template · type on screen · import any other file (pick ID column + subject columns, marks out of any total)
- 📘 Homework: create homework, mark Submitted / Late / Missing, see how homework links to marks

**Deeper analysis**
- 🔔 Early Warnings: absent 3+ days in a row, low 30-day attendance, 3 missed homeworks — call list for class teachers
- 🎯 Targets & Progress: personal next-exam target for every student from their own history; Class 8 → now tracking
- ⚖️ Teacher Workload: periods/week, class sizes, other duties vs results
- 🔍 Mistake Analysis: question-wise marks → careless mistakes vs concept gaps vs hard-problem practice
- Prediction accuracy checked honestly (Pilot Toolkit tab 4)

**AI (🤖 AI Assistant)** — "Ask the data" chat, kind teacher feedback, remedial worksheets for weak topics, report-card remarks in English + Telugu (on Report Cards page).
Works offline with built-in rules/templates. To use Claude AI:
```
export ANTHROPIC_API_KEY=sk-ant-...      # or copy .streamlit/secrets.toml.example → secrets.toml
export CLAUDE_MODEL=claude-sonnet-4-5    # optional; use any current Claude model id
```
The chat agent can only call fixed analysis tools on your data, so numbers come from the data, not from the AI. Always read AI text before sending it to anyone.

## New in version 4 — stronger, more honest analysis
1. **Absences handled correctly** — a missed paper is no longer counted as 0 (it was pulling averages down and could wrongly blame teachers). **🩺 Data Health** page checks every upload: impossible marks, unknown IDs, absences, sudden jumps, copied-looking marks.
   **⚖️ Adjust for exam difficulty** (sidebar switch) — removes the effect of one paper being harder for everyone (Unit Test 2 in the sample).
2. **Uncertainty and size of differences** — every section comparison shows a likely range and says whether the gap is *small / medium / large*, corrected for testing many sections at once. Alerts appear only when a gap is both real and big enough to matter.
3. **Why is this student at risk?** — SHAP shows which facts raise or lower each student's risk, plus a short explanation a teacher can read at a parent meeting. **Fairness check** shows whether the warning system is equally accurate for boys/girls, internet/no internet, sections, etc.
4. **Honest Action Tracker** — each targeted student is compared with the most similar student who did *not* get the action, so the natural bounce-back of weak students is not counted as success. Shows a likely range.
5. **Fair teacher comparison (multilevel model)** — accounts for students' last-year results, attendance and homework; small classes are pulled toward the average so one unlucky group doesn't label a teacher. Same for each class (subject × section).

## Data (data/)
students, teachers, marks (with `absent` column: 1 = absent), attendance (monthly), daily_attendance, homework, homework_status, history (past years), questions + question_marks (question-wise Final), sections (class teachers), last_year, actions, targets.json, users.
The included data is SAMPLE data created by generate_data.py — running it again resets everything (including passwords and actions).
