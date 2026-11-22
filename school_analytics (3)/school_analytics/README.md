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

## New in version 5 — Agentic AI 🧠
Agents that **watch the data, decide what matters and prepare the work — a person approves everything.** Open **🧠 AI Agents** in the app, or run them on a schedule with `run_agents.py`.

| Agent | What it does on its own | Screenshot |
|---|---|---|
| 📰 **Weekly briefing** | Checks attendance, early warnings, homework, risk, actions and data quality → "the 3 things to act on this week" as Problem → Why → Action → Owner → Deadline | [01](docs/agent_screenshots/01_weekly_briefing.png) |
| ✅ **Action review** | After each exam measures every open action against similar students → **close / extend / change approach**, with a drafted follow-up action | [02](docs/agent_screenshots/02_action_review.png) |
| 🔔 **Early-warning drafts** | 3+ days absent (or <65% in 30 days) → parent message in **English + Telugu** and a call note for the class teacher | [03](docs/agent_screenshots/03_early_warning_drafts.png) |
| 🩹 **Remedial planner** | Mistake diagnosis → 2-week day-by-day plan + 2 worksheets → **re-checks after the next exam** and adapts. Can pick the highest-risk students itself | [04](docs/agent_screenshots/04_remedial_planner.png) |
| 🧪 **Data checks** | Impossible marks, mistyped IDs, sudden jumps → a question + suggested fix for the teacher who entered them | [05](docs/agent_screenshots/05_data_checks.png) |
| 💬 **Chat with more tools** | `early_warnings`, `homework_summary`, `mistake_profile`, `student_targets`, `fair_teacher_effects`, `explain_student` (SHAP), `attendance_summary` — ask *"Why is S1479 at risk and what should we do?"* | [08](docs/agent_screenshots/08_chat_why_is_student_at_risk.png) |

Also: [approval inbox](docs/agent_screenshots/06_approval_inbox.png), [audit log](docs/agent_screenshots/07_agent_log.png), and the [teacher's view](docs/agent_screenshots/09_teacher_inbox.png) (a teacher sees only their own items).

### Safety rules (built in and tested)
- **Nothing is sent and no marks change until a person presses Approve.** Approving a data fix is the only thing that edits marks; approving an action proposal updates the Action Tracker.
- Every agent run and every human decision is written to `agent_log.csv`; drafts wait in `agent_queue.csv`. Rejected/approved items are not re-created for 7 days (no nagging).
- **Claude never sees student names.** Messages are written with `{STUDENT}` placeholders and filled in locally; chat tool results drop the name column; the new tools return IDs only.
- **Grounding check:** if Claude's briefing quotes a number that is not in the tool results, it is thrown away and the built-in version is used. If Claude is unavailable, everything still works offline.
- Role scope: teachers see only their items/students; HODs their subject; parents none.
- Remedial plans and parent messages are drafts — read them before use. Add school consent before sending any student data to an external AI service.

### Run on a schedule
```
python run_agents.py daily        # early-warning drafts + data checks   (every school morning)
python run_agents.py weekly       # Principal briefing                   (Monday)
python run_agents.py after-exam   # action review, remedial plans, plan re-check
python run_agents.py all --no-llm
```
Outputs go to `data/agent_outputs/`. Example cron: `30 7 * * 1  cd /path/to/school_analytics && python run_agents.py weekly`.
Set `SCHOOL_NAME` to put your school's name in parent messages, and `SCHOOL_DATA_DIR` to point at another data folder.

### Demo data & tests
`python make_demo_data.py` makes `demo_data/` (a copy of `data/` with a few realistic problems added — a mistyped ID, impossible marks, a failed action — used for the screenshots). `SCHOOL_DATA_DIR=demo_data streamlit run app.py`. Tests: `pip install pytest && pytest -q tests` (Claude paths are tested with a fake client, so no API key or network is needed).

## Layer 1 — free "investigator" in the AI Assistant (no AI model, no key, ₹0)
With no `ANTHROPIC_API_KEY`, **Ask the data** no longer just matches keywords. `planner.py` runs a small multi-step investigation: it calls a tool, **reads the result, and a rule decides what to look at next**.

Example — *"Why are Maths marks low?"*
1. `subject_summary` → Maths is the lowest subject
2. `section_comparison` → Section C is 14 marks behind Section A → **the problem is one class** (if all sections were alike it would instead check the kind of mistakes)
3. `fair_teacher_effects` → is it the teacher, after allowing for students' past marks?
4. `topic_scores(section=C)` → weakest topic
5. `homework_summary` → is homework a factor?

Playbooks: weakest subject / "why is X low", who is at risk, absences, homework, teachers who need support, careless vs concept gaps, trends, and one student (`S1479` — skips the attendance-alert check when attendance is fine).
Every answer ends with a **🔎 How I investigated** list of the steps and what each found. Screenshot: [10](docs/agent_screenshots/10_investigator_maths.png).

**Guarantees (tested):** never repeats the same call, stops at 10 steps, skips checks that do not apply, degrades gracefully if data is missing, respects role scope, and **never calls a model** (the tests fail if it does).
**Limits:** it is a rule-based script, not AI. It only handles the question types above; anything else gets the simple subject summary. It can't reason about open questions — for those, add a Claude key (see above).
