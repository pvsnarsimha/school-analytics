"""School Performance Analytics — Streamlit app with role-based logins.
Run:  streamlit run app.py
"""
import os, json, hashlib, datetime as dt
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import analytics as A
import reports as R
import analytics2 as B
import analytics3 as C3
import ai
import agents as AG

st.set_page_config(page_title="School Performance Analytics", page_icon="📊", layout="wide")
DATA = os.environ.get("SCHOOL_DATA_DIR") or os.path.join(os.path.dirname(__file__), "data")
P = lambda f: os.path.join(DATA, f)
SEQ = ["#4f8ff7", "#f2c811", "#2fbf71", "#e86bd1", "#22c3d6", "#f58a42"]
STATUS_COLOR = {"Good": "#2fbf71", "Watch": "#f5a524", "Action needed": "#ef5b5b"}
hash_pw = lambda pw: hashlib.sha256(("school-salt:" + pw).encode()).hexdigest()

PRINT = st.session_state.get("print_mode", False)
px.defaults.template = "plotly_white" if PRINT else "plotly_dark"
px.defaults.color_discrete_sequence = SEQ

BASE_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600;700&display=swap');
html,body,.stApp,.stMarkdown,[data-testid="stMetricLabel"],[data-testid="stMetricValue"],.js-plotly-plot text{font-family:"Source Sans Pro","Source Sans 3",sans-serif,"Noto Sans Telugu","Nirmala UI","Gautami"!important}
.block-container{padding-top:1.3rem}
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-left:4px solid #4f8ff7;border-radius:10px;padding:12px 14px}
.find{padding:11px 15px;border-radius:8px;background:var(--card2);margin-bottom:9px;border-left:5px solid #4f8ff7;line-height:1.5}
.find.red{border-color:#ef5b5b}.find.amb{border-color:#f5a524}.find.grn{border-color:#2fbf71}
.headline{font-size:1.15rem;font-weight:600;padding:14px 18px;border-radius:10px;background:var(--card2);border:1px solid var(--line);margin-bottom:14px}
.pp{display:grid;grid-template-columns:150px 1fr;gap:3px 12px;font-size:.95rem}.pp b{color:var(--mut)}
.goal{display:flex;align-items:center;gap:12px;margin:6px 0}.goal .lab{width:220px}.goal .trk{flex:1;height:14px;border-radius:7px;background:var(--line)}
.goal .fil{height:100%;border-radius:7px}.goal .val{width:260px;font-size:.9rem;color:var(--mut)}
:root{--card:#171e2e;--card2:#1d2639;--line:#2a344a;--mut:#9aa6bd}
</style>"""
PRINT_CSS = """<style>
:root{--card:#ffffff!important;--card2:#f4f6fb!important;--line:#dde2ec!important;--mut:#5a6478!important}
div[data-testid="stMetric"]{background:#fff!important;border:1px solid #dde2ec!important}
.find,.headline{background:#f4f6fb!important}
.stButton button{background:#fff!important;border:1px solid #888!important}.stButton button *{color:#111!important}
[data-testid="stExpander"] details{background:#fff!important}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#fff!important;color:#111!important}
.stApp *{color:#111}
section[data-testid="stSidebar"],[data-testid="stToolbar"],header{display:none!important}
.block-container{max-width:1100px}
@media print{.noprint,.stButton,.stDownloadButton{display:none!important}}
</style>"""
st.markdown(BASE_CSS + (PRINT_CSS if PRINT else ""), unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def card(text, kind=""):
    st.markdown(f'<div class="find {kind}">{text}</div>', unsafe_allow_html=True)


def explain(text):
    with st.expander("ℹ️ What does this mean?"):
        st.write(text)


def fig_style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.2, title=None))
    return fig


def show(fig, h=360):
    if PRINT:
        fig.update_layout(font_color="#111", title_font_color="#111")
    st.plotly_chart(fig_style(fig, h), use_container_width=True, theme=None if PRINT else "streamlit")


def goal_bar(label, now, target, unit=""):
    pct = min(100, now / target * 100)
    col = "#2fbf71" if pct >= 100 else "#f5a524" if pct >= 90 else "#ef5b5b"
    st.markdown(f'<div class="goal"><div class="lab">{label}</div><div class="trk"><div class="fil" style="width:{pct}%;background:{col}"></div></div>'
                f'<div class="val">{now:.1f}{unit} of {target}{unit} target ({pct:.0f}%)</div></div>', unsafe_allow_html=True)


def mtime(*fs):
    return tuple(os.path.getmtime(P(f)) if os.path.exists(P(f)) else 0 for f in fs)


# ------------------------------------------------------------------ data
@st.cache_data
def load(_key):
    rd = lambda f: pd.read_csv(P(f))
    return (rd("students.csv"), rd("teachers.csv"), rd("marks.csv"), rd("attendance.csv"),
            rd("last_year.csv") if os.path.exists(P("last_year.csv")) else None)


@st.cache_data
def scores(_marks, key):
    return A.subject_scores(_marks)


@st.cache_data
def run_risk(_ss, _students, _attendance, key):
    prob, rs, met, imp, X = A.risk_model(_ss, _students, _attendance)
    return prob, rs.to_dict(), met, imp


@st.cache_data
def load_extra(_key):
    rd = lambda f: pd.read_csv(P(f)) if os.path.exists(P(f)) else None
    return {n: rd(f"{n}.csv") for n in ["daily_attendance", "sections", "homework", "homework_status", "history", "questions", "question_marks"]}


@st.cache_data
def hw_tables(_hw, _hws, key):
    return B.homework_tables(_hw, _hws)


@st.cache_data
def targets_table(_ss, _hist, key):
    return B.student_targets(_ss, _hist)


@st.cache_data
def mistakes(_qm, _qs, key):
    return B.mistake_profile(_qm, _qs)


@st.cache_data
def risk_explained(_ss, _students, _att, key):
    return C3.explain_risk(_ss, _students, _att)


@st.cache_data
def fairness(_ss, _students, _att, key):
    return C3.fairness_check(_ss, _students, _att)


@st.cache_data
def fair_teachers(_ss, _students, _teachers, exam, key):
    return C3.fair_teacher_effects(_ss, _students, _teachers, exam)


@st.cache_data
def fair_classes(_ss, _students, exam, key):
    return C3.fair_class_effects(_ss, _students, exam)


VERDICT_COLOR = {"Below expected": "#ef5b5b", "Slightly below": "#f5a524", "As expected": "#8f9bb3", "Slightly above": "#7fd6a4", "Above expected": "#2fbf71"}


def save_marks(long, ex, ex_order, subj, t):
    """Replace this exam+subject's rows for these students and save."""
    long = long.assign(exam=ex, exam_order=ex_order, subject=subj, teacher_id=t, max_marks=25)
    if "absent" not in long: long["absent"] = 0
    keep = ~((marks.exam == ex) & (marks.subject == subj) & marks.student_id.isin(long.student_id))
    pd.concat([marks[keep], long[marks.columns]], ignore_index=True).to_csv(P("marks.csv"), index=False)
    st.cache_data.clear()


def load_actions():
    if os.path.exists(P("actions.csv")):
        return pd.read_csv(P("actions.csv"), dtype={"target_students": str}).fillna({"target_students": ""})
    return pd.DataFrame(columns=["action_id", "created", "problem", "reason", "action", "owner", "deadline", "status", "subject", "start_exam_order", "target_students"])


def save_actions(df):
    df.to_csv(P("actions.csv"), index=False)


def load_targets():
    return json.load(open(P("targets.json")))


def users():
    return pd.read_csv(P("users.csv"), dtype=str)


KEY = mtime("students.csv", "teachers.csv", "marks.csv", "attendance.csv", "last_year.csv")
students, teachers, marks, attendance, last_year = load(KEY)
ss = scores(marks, KEY)
EXAMS = ss.sort_values("exam_order").exam.unique().tolist()
SUBJECTS = sorted(ss.subject.unique())
TOPICS = marks.groupby("subject").topic.unique().apply(list).to_dict()
tname = teachers.set_index("teacher_id").teacher_name.to_dict()
tsubj = teachers.set_index("teacher_id").subject.to_dict()
targets = load_targets()
XKEY = mtime("daily_attendance.csv", "homework_status.csv", "homework.csv", "history.csv", "question_marks.csv")
X = load_extra(XKEY)
sections = X["sections"] if X["sections"] is not None else pd.DataFrame({"section": students.section.unique(), "class_teacher_id": None})
hw_join, hw_per = hw_tables(X["homework"], X["homework_status"], XKEY) if X["homework"] is not None else (None, None)
risk_prob, risk_reasons, risk_metrics, risk_imp = run_risk(ss, students, attendance, KEY) if len(EXAMS) >= 2 else (pd.Series(0, index=students.student_id), {}, {}, pd.Series(dtype=float))

# ------------------------------------------------------------------ login
if "user" not in st.session_state:
    l, c, r = st.columns([1, 1.3, 1])
    with c:
        st.markdown("## 📊 School Performance Analytics")
        st.caption("Sign in to see the view for your role.")
        with st.form("login"):
            u = st.text_input("Username").strip().lower()
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                us = users(); m = us[us.username == u]
                if len(m) and m.iloc[0].password_hash == hash_pw(pw):
                    st.session_state.user = m.iloc[0].to_dict(); st.rerun()
                else:
                    st.error("Wrong username or password.")
        with st.expander("Demo accounts"):
            st.markdown("""| Role | Username | Password |
|---|---|---|
| Principal | `principal` | `principal123` |
| HOD | `hod_mathematics`, `hod_physics`, … | `hod123` |
| Teacher | `t01` … `t19` (t19 = Section C Maths) | `teacher123` |
| Parent | student ID, e.g. `s1085` | `parent123` |""")
    st.stop()

U = st.session_state.user
ROLE, SCOPE = U["role"], U["scope"]

# ------------------------------------------------------------------ scope: what this user may see
if ROLE == "teacher":
    my_pairs = ss[ss.teacher_id == SCOPE][["student_id"]].drop_duplicates()
    VISIBLE = students[students.student_id.isin(my_pairs.student_id)]
elif ROLE == "parent":
    VISIBLE = students[students.student_id == SCOPE]
else:
    VISIBLE = students

PAGES = {
    "principal": ["🏠 Principal Summary", "🔔 Early Warnings", "🏫 School Details", "📚 Subjects & Topics", "👩‍🏫 Teachers", "⚖️ Teacher Workload",
                  "⚠️ Students at Risk", "🔍 Mistake Analysis", "🎯 Targets & Progress", "🔬 Deep Analysis", "🤖 AI Assistant", "🧠 AI Agents", "✅ Action Tracker",
                  "🎓 Report Cards", "📄 PDF Reports", "📅 Attendance", "📘 Homework", "📝 Marks Entry", "🧪 Pilot Toolkit", "🩺 Data Health", "⚙️ Settings"],
    "hod": ["📚 Subjects & Topics", "🔔 Early Warnings", "👩‍🏫 Teachers", "⚖️ Teacher Workload", "⚠️ Students at Risk", "🔍 Mistake Analysis",
            "🎯 Targets & Progress", "🤖 AI Assistant", "🧠 AI Agents", "✅ Action Tracker", "🎓 Report Cards", "📄 PDF Reports", "📘 Homework", "📝 Marks Entry", "🩺 Data Health", "⚙️ Settings"],
    "teacher": ["🔔 Early Warnings", "📅 Attendance", "👩‍🏫 Teachers", "⚠️ Students at Risk", "🔍 Mistake Analysis", "🎯 Targets & Progress",
                "🤖 AI Assistant", "🧠 AI Agents", "🎓 Report Cards", "📄 PDF Reports", "📘 Homework", "📝 Marks Entry", "⚙️ Settings"],
    "parent": ["🎓 Report Cards", "⚙️ Settings"],
}[ROLE]

# ------------------------------------------------------------------ sidebar / print bar
if PRINT:
    c1, c2 = st.columns([5, 1])
    c2.button("⬅ Exit print mode", on_click=lambda: st.session_state.update(print_mode=False))
    c1.caption("Print mode — use your browser's Print (Ctrl+P) to print or save as PDF.")
    page = st.session_state.get("page", PAGES[0])
    exam = st.session_state.get("exam", EXAMS[-1])
else:
    st.sidebar.markdown(f"### 📊 School Analytics\n**{U['display_name']}**  \n<span style='color:#9aa6bd'>{ROLE.title()}</span>", unsafe_allow_html=True)
    page = st.sidebar.radio("Go to", PAGES, key="page")
    exam = st.sidebar.selectbox("Exam", EXAMS, index=len(EXAMS) - 1, key="exam")
    st.sidebar.toggle("⚖️ Adjust for exam difficulty", key="adjust", help="Removes the effect of one exam paper being harder or easier for everyone, so trends show real change.")
    st.sidebar.toggle("🖨️ Print-friendly mode", key="print_mode")
    if st.sidebar.button("Sign out"):
        st.session_state.clear(); st.rerun()
    st.sidebar.caption(f"Pass mark {A.PASS} · {len(students)} students · {len(EXAMS)} exams")

EI = EXAMS.index(exam)
ADJ = st.session_state.get("adjust", False)
SC = "adj_score" if ADJ else "score"
ADJ_NOTE = " (adjusted for exam difficulty)" if ADJ else ""
e_now = ss[ss.exam == exam].merge(students[["student_id", "section"]], on="student_id")

# ================================================================== PRINCIPAL SUMMARY
if page == "🏠 Principal Summary":
    st.title("Principal Summary")
    st.caption(f"{exam} — everything important on one page. Biggest problem first.")
    status = A.subject_status(ss, students, exam, targets, last_year)
    probs = A.principal_problems(ss, students, teachers, marks, exam, risk_prob if EI == len(EXAMS) - 1 else None)
    k = A.exam_kpis(ss, students, exam); kp = A.exam_kpis(ss, students, EXAMS[EI - 1]) if EI else None
    st.markdown(f'<div class="headline">📌 {A.headline(status, probs, k, kp)}</div>', unsafe_allow_html=True)

    st.subheader("Subject health")
    cols = st.columns(len(status))
    for c, r in zip(cols, status.itertuples()):
        c.markdown(f"""<div style="background:var(--card);border:1px solid var(--line);border-top:5px solid {STATUS_COLOR[r.status]};
        border-radius:10px;padding:10px 12px"><div style="font-size:.85rem;color:var(--mut)">{r.subject}</div>
        <div style="font-size:1.7rem;font-weight:700">{r.avg:.1f}<span style="font-size:.8rem;font-weight:400;color:var(--mut)"> ±{(A.mean_ci(e_now[e_now.subject == r.subject].score)[2] - r.avg):.1f}</span></div>
        <div style="font-size:.8rem">{A.STATUS_ICON[r.status]} {r.status}</div>
        <div style="font-size:.75rem;color:var(--mut)">vs last exam: {A.change_words(r.vs_last_exam)}<br>vs last year: {A.change_words(r.vs_last_year)}</div></div>""",
                   unsafe_allow_html=True)
    explain("Each box is one subject's average out of 100 (± shows how much it could move by chance). 🟢 Good = on target. 🟡 Watch = a little below target or a few students failing. "
            "🔴 Action needed = well below target or many students failing. 'vs last year' compares with last year's batch in the same exam.")

    st.subheader("Top 3 problems — and what to do")
    acts = load_actions()
    for i, p in enumerate(probs[:3]):
        kind = "red" if p["severity"] == 3 else "amb"
        card(f"""<div style="font-size:1.05rem;font-weight:700;margin-bottom:6px">{i + 1}. {p['problem']}</div><div class="pp">
        <b>Why</b><span>{p['reason']}</span><b>Action</b><span>{p['action']}</span>
        <b>Owner</b><span>{p['owner']}</span><b>Deadline</b><span>{p['deadline']}</span></div>""", kind)
        already = (acts.problem == p["problem"]).any()
        if not PRINT and st.button("✅ Already in Action Tracker" if already else "➕ Add to Action Tracker", key=f"addp{i}", disabled=already):
            if p["subject"] != "All":
                tgt = e_now[(e_now.subject == p["subject"]) & (e_now.score < 50)].student_id
            else:
                tgt = risk_prob[risk_prob >= .6].index
            new = {"action_id": int(acts.action_id.max() + 1) if len(acts) else 1, "created": dt.date.today().isoformat(),
                   "problem": p["problem"], "reason": p["reason"], "action": p["action"], "owner": p["owner"],
                   "deadline": p["deadline"], "status": "Not started", "subject": p["subject"] if p["subject"] != "All" else "",
                   "start_exam_order": int(ss.exam_order.max()), "target_students": ";".join(tgt)}
            save_actions(pd.concat([acts, pd.DataFrame([new])], ignore_index=True)); st.rerun()

    st.subheader("Are we reaching our goals?")
    for r in A.target_progress(ss, students, exam, targets).itertuples():
        goal_bar(r.Goal, r.Now, r.Target, r.unit)
    explain("Targets are set by the Principal in Settings. A full bar means the target is reached.")

    st.subheader("Last action results")
    for a in acts.to_dict("records")[-3:]:
        im = C3.matched_impact(ss, students, a)
        if im:
            card(f"<b>{im['verdict']} — {a['action']}</b> ({a['owner']})<br>{im['sentence']}", "grn" if im["worked"] else "amb")
        else:
            card(f"<b>{a['action']}</b> — {a['status']}. Result will show after the next exam.", "")

# ================================================================== SCHOOL DETAILS
elif page == "🏫 School Details":
    st.title("School Details")
    k = A.exam_kpis(ss, students, exam); kp = A.exam_kpis(ss, students, EXAMS[EI - 1]) if EI else None
    d = lambda key, f="{:+.1f}": f.format(k[key] - kp[key]) if kp else None
    c = st.columns(4)
    c[0].metric("School average", f"{k['avg']:.1f}", d("avg"), help="Average of all students in all subjects, out of 100")
    c[1].metric("Passed all subjects", f"{k['pass_rate']:.1f}%", d("pass_rate"))
    c[2].metric("Distinction (75+)", f"{k['distinction']:.1f}%", d("distinction"))
    c[3].metric("Failed 1 or more subjects", k["failing"], d("failing", "{:+.0f}"), delta_color="inverse")
    l, r = st.columns(2)
    with l:
        hm = e_now.pivot_table(index="section", columns="subject", values="score", aggfunc="mean")
        show(px.imshow(hm.round(0), text_auto=True, color_continuous_scale="RdYlGn", zmin=45, zmax=75, aspect="auto",
                       title="1. Where is the problem? (section × subject average)"))
        gaps = A.section_gaps(ss, students, exam)
        act = gaps[gaps.worth_acting & (gaps["diff"] < 0)]
        for g_ in act.itertuples():
            card(f"<b>{g_.subject}, Section {g_.section}</b>: {abs(g_.diff):.1f} marks below other sections (likely {abs(g_.hi):.1f}–{abs(g_.lo):.1f}). "
                 f"A <b>{g_.size}</b> difference that is real, not chance — worth acting on.", "red")
        n_small = int((gaps.real & ~gaps.worth_acting).sum())
        if n_small: st.caption(f"{n_small} other section differences are real but small — not worth special action.")
        explain("Read across a row. If only one box is red, that subject's teaching in that section is the problem. "
                "If a whole column is red, the subject is hard for all students. We only raise an alert when the gap is both real (not luck, even after checking 24 sections at once) and big enough to matter.")
    with r:
        tr = ss.groupby(["exam_order", "exam", "subject"])[SC].mean().reset_index()
        show(px.line(tr, x="exam", y=SC, color="subject", markers=True, title="2. Are subjects improving or falling?" + ADJ_NOTE, labels={SC: "Average"}))
        dif = A.exam_difficulty(ss); hard = dif[dif.verdict != "Normal"]
        if len(hard) and not ADJ:
            card(f"⚠️ <b>{hard.iloc[0].exam}</b> was a {hard.iloc[0].verdict.lower()} ({hard.iloc[0]['shift']:+.1f} marks for the whole school). "
                 "Part of the drop in this chart is the paper, not the students. Turn on <b>⚖️ Adjust for exam difficulty</b> in the sidebar to see the real trend.", "amb")
        explain("Each line is one subject across the year's exams. A falling line is a warning — but first check whether that exam paper was simply harder for everyone.")
    l, r = st.columns(2)
    with l:
        ab = pd.cut(students.attendance_pct, [0, 65, 75, 85, 101], labels=["Below 65%", "65–75%", "75–85%", "85%+"], right=False)
        sa = e_now.groupby("student_id").score.mean()
        g = pd.DataFrame({"band": ab.values, "avg": sa.reindex(students.student_id).values}).groupby("band", observed=True).avg.agg(["mean", "size"]).reset_index()
        show(px.bar(g, x="band", y="mean", text=g["mean"].round(0), title="3. Attendance and marks", labels={"mean": "Average score", "band": "Attendance"},
                    color="mean", color_continuous_scale="RdYlGn", range_color=[45, 70]).update_layout(coloraxis_showscale=False))
        explain("Students who attend more score more. This is the single most common reason for low marks.")
    with r:
        if last_year is not None and EI == len(EXAMS) - 1:
            ly = last_year.groupby("subject").avg_score.mean()
            cmp = pd.DataFrame({"This year": e_now.groupby("subject").score.mean(), "Last year": ly}).reset_index().melt(id_vars="subject")
            show(px.bar(cmp, x="subject", y="value", color="variable", barmode="group", title="4. This year vs last year (final exam)",
                        color_discrete_map={"This year": "#4f8ff7", "Last year": "#8f9bb3"}, labels={"value": "Average"}))
            explain("Compares this year's final with last year's batch. Taller blue bar = better than last year.")
        else:
            st.info("Year-on-year comparison is available for the final exam.")

# ================================================================== SUBJECTS & TOPICS (HOD)
elif page == "📚 Subjects & Topics":
    subj_list = [SCOPE] if ROLE == "hod" else SUBJECTS
    st.title("Subjects & Topics")
    subj = st.selectbox("Subject", subj_list) if len(subj_list) > 1 else subj_list[0]
    s = e_now[e_now.subject == subj]
    stt = A.status_of(s.score.mean(), (s.score < A.PASS).mean() * 100, targets["subject_avg"].get(subj, 60))
    c = st.columns(4)
    c[0].metric("Subject average", f"{s.score.mean():.1f}", help=f"Target {targets['subject_avg'].get(subj, 60)}")
    c[1].metric("Students failing", int((s.score < A.PASS).sum()))
    c[2].metric("Distinctions", int((s.score >= 75).sum()))
    c[3].metric("Status", f"{A.STATUS_ICON[stt]} {stt}")
    m = marks[(marks.subject == subj) & (marks.exam == exam)].merge(students[["student_id", "section"]], on="student_id")
    l, r = st.columns(2)
    with l:
        hm = m.pivot_table(index="topic", columns="section", values="marks", aggfunc="mean") / 25 * 100
        show(px.imshow(hm.round(0), text_auto=True, color_continuous_scale="RdYlGn", zmin=35, zmax=80, aspect="auto",
                       title="Which topics need re-teaching? (% scored)"))
        weak = hm.mean(1).sort_values()
        one_sec = (hm.loc[weak.index[0]].max() - hm.loc[weak.index[0]].min()) > 10
        card(f"<b>Weakest topic: {weak.index[0]}</b> ({weak.iloc[0]:.0f}% scored). " +
             ("It is much weaker in some sections — support those teachers." if one_sec else "It is weak in every section — revise the lesson plan and teaching method."), "red")
    with r:
        w = m.pivot_table(index="student_id", columns="topic", values="marks")
        tot = w.sum(1); hi, lo = tot >= tot.quantile(.73), tot <= tot.quantile(.27)
        item = pd.DataFrame({"Avg scored %": w.mean() / 25 * 100, "Separates strong/weak": (w[hi].mean() - w[lo].mean()) / 25}).reset_index()
        item["Verdict"] = np.where(item["Separates strong/weak"] < .2, "Revise this question section", np.where(item["Separates strong/weak"] < .3, "OK", "Good question"))
        show(px.scatter(item, x="Avg scored %", y="Separates strong/weak", text="topic", color="Verdict", size=[18] * len(item),
                        title="Is the question paper good?", color_discrete_map={"Good question": "#2fbf71", "OK": "#f5a524", "Revise this question section": "#ef5b5b"}).update_traces(textposition="top center"))
        explain("A good question gives high marks to strong students and low marks to weak ones. "
                "If strong and weak students score about the same on a section, that part of the paper does not test understanding — revise it.")
    st.subheader("Is the gap between sections real — and does it matter?")
    gp = A.section_gaps(ss, students, exam)
    gp = gp[gp.subject == subj].copy()
    gp["label"] = "Section " + gp.section
    fig = go.Figure(go.Scatter(x=gp["diff"], y=gp.label, mode="markers", marker=dict(size=14, color=[VERDICT_COLOR["Below expected"] if (r_.real and r_.diff < 0 and r_.worth_acting) else
                    VERDICT_COLOR["Above expected"] if (r_.real and r_.diff > 0 and r_.worth_acting) else "#8f9bb3" for r_ in gp.itertuples()]),
                    error_x=dict(type="data", symmetric=False, array=gp.hi - gp["diff"], arrayminus=gp["diff"] - gp.lo, thickness=2)))
    fig.add_vline(x=0, line_dash="dash", line_color="#888")
    fig.update_layout(title="Each section compared with the others (dot = difference, line = likely range)", xaxis_title="Marks above (+) / below (−) other sections")
    show(fig, 300)
    for r_ in gp.itertuples():
        if r_.worth_acting:
            card(f"<b>Section {r_.section}</b> is {abs(r_.diff):.1f} marks {'below' if r_.diff < 0 else 'above'} the other sections (likely between {min(abs(r_.lo), abs(r_.hi)):.1f} and {max(abs(r_.lo), abs(r_.hi)):.1f}). "
                 f"This is a <b>{r_.size}</b>, real difference.", "red" if r_.diff < 0 else "grn")
    if not gp.worth_acting.any():
        card("No section is different enough to need special action. Small gaps are normal.", "grn")
    explain("The dot shows the difference; the line shows the range it is likely to be in. If the line crosses the dashed zero line, the gap could be chance. "
            "'Small / medium / large' tells you whether it is big enough to matter — a real 1-mark gap is not worth acting on.")
    with st.expander("Show technical details"):
        st.dataframe(gp[["section", "average", "diff", "lo", "hi", "d", "size", "p", "p_adjusted", "verdict"]].round(3), hide_index=True, use_container_width=True)
        st.caption("Welch t-test of each section vs the rest; 95% confidence interval; Cohen's d; p-values corrected for testing many sections at once (Benjamini–Hochberg).")

# ================================================================== TEACHERS
elif page == "👩‍🏫 Teachers":
    if ROLE == "teacher":
        t = SCOPE; st.title("My Classes")
    else:
        tl = teachers[teachers.teacher_id.isin(ss.teacher_id.unique())]
        if ROLE == "hod": tl = tl[tl.subject == SCOPE]
        st.title("Teachers")
        if ROLE == "principal":
            fe, info = fair_teachers(ss, students, teachers, exam, KEY)
            fe["label"] = fe.teacher_name + " · " + fe.subject
            fig = px.scatter(fe, x="effect", y="label", color="verdict", color_discrete_map=VERDICT_COLOR, error_x=fe.hi - fe.effect, error_x_minus=fe.effect - fe.lo,
                             title="Fair comparison: marks above / below what we expect for students like these",
                             labels={"effect": "Marks above (+) / below (−) expected", "label": ""}, hover_data={"students": True, "raw_gap": ":.1f"})
            fig.add_vline(x=0, line_dash="dash", line_color="#888"); fig.update_traces(marker_size=11)
            show(fig, 600)
            card(f"The background of the students (subject, last year's result, attendance, homework) explains about <b>{info['r2_background'] * 100:.0f}%</b> of the differences in marks. "
                 f"Who teaches the class explains a further part — teachers typically differ by about <b>±{info['between_group_sd']:.1f} marks</b>.", "")
            explain("A teacher who gets weaker students should not look bad because of it. So we first predict each student's marks from their background, then see whether the class "
                    "did better or worse than predicted. Small classes are pulled toward the average so one unlucky group doesn't unfairly label a teacher. "
                    "The line shows the likely range. Use this to start a supportive conversation — and check ⚖️ Teacher Workload before judging.")
            with st.expander("Show technical details"):
                st.dataframe(fe[["teacher_name", "subject", "students", "raw_gap", "unshrunk", "effect", "lo", "hi", "reliability", "verdict"]].round(2), hide_index=True, use_container_width=True)
                st.caption("Random-intercept (multilevel) model estimated by empirical Bayes: fixed effects for subject, previous-year %, attendance and homework; "
                           f"teacher effects shrunk by reliability. Between-teacher SD {info['between_group_sd']:.2f}, student SD {info['student_sd']:.2f}, ICC {info['icc']:.2f}.")
        t = st.selectbox("Select teacher", tl.teacher_id, format_func=lambda x: f"{tname[x]} — {tsubj[x]}")
    subj = tsubj[t]
    mine = e_now[e_now.teacher_id == t]; school = e_now[e_now.subject == subj]
    st.caption(f"{tname[t]} · {subj} · Sections {', '.join(sorted(mine.section.unique()))} · {mine.student_id.nunique()} students · {exam}")
    fe_all, _ = fair_teachers(ss, students, teachers, exam, KEY); fe_t = fe_all.set_index("teacher_id")
    c = st.columns(4)
    c[0].metric("Class average", f"{mine.score.mean():.1f}", f"{mine.score.mean() - school.score.mean():+.1f} vs school")
    c[1].metric("Failing", int((mine.score < A.PASS).sum()))
    c[2].metric("Distinctions (75+)", int((mine.score >= 75).sum()))
    if t in fe_t.index:
        r_t = fe_t.loc[t]
        c[3].metric("Compared with expected", f"{r_t.effect:+.1f} marks", r_t.verdict, delta_color="off",
                    help=f"Likely between {r_t.lo:+.1f} and {r_t.hi:+.1f}. Allows for students' last-year results, attendance and homework.")
        card(C3.explain_effect(r_t, "your" if ROLE == "teacher" else tname[t]).replace("your's", "your").replace("**", ""),
             "red" if "below" in r_t.verdict.lower() else "grn" if "above" in r_t.verdict.lower() else "")
    tm = marks[marks.subject == subj].assign(who=lambda x: np.where(x.teacher_id == t, "My classes", "Rest of school"))
    l, r = st.columns(2)
    with l:
        tp = tm[tm.exam == exam].groupby(["topic", "who"]).marks.mean().div(25).mul(100).reset_index()
        show(px.bar(tp, x="topic", y="marks", color="who", barmode="group", title="Where do my students lose marks?",
                    color_discrete_map={"My classes": "#f2c811", "Rest of school": "#4f8ff7"}, labels={"marks": "% scored"}))
    with r:
        tr = ss[ss.subject == subj].assign(who=lambda x: np.where(x.teacher_id == t, "My classes", "Rest of school")).groupby(["exam_order", "exam", "who"])[SC].mean().reset_index()
        show(px.line(tr, x="exam", y=SC, color="who", markers=True, title="My class vs school across exams" + ADJ_NOTE, labels={SC: "Average"},
                     color_discrete_map={"My classes": "#f2c811", "Rest of school": "#4f8ff7"}))
    weakest = (tm[(tm.exam == exam) & (tm.who == "My classes")].groupby("topic").marks.mean() / 25 * 100).sort_values()
    card(f"<b>Re-teach first:</b> {weakest.index[0]} ({weakest.iloc[0]:.0f}% scored) and {weakest.index[1]} ({weakest.iloc[1]:.0f}%).", "red")
    st.subheader("Students who need attention")
    lst = mine[["student_id", "section", "score"]].merge(students[["student_id", "student_name", "attendance_pct", "assignment_submission_pct"]], on="student_id")
    first = ss[(ss.subject == subj) & (ss.exam_order == 1)].set_index("student_id").score
    lst["change"] = lst.score - lst.student_id.map(first)
    lst["risk"] = (lst.student_id.map(risk_prob) * 100).round(0)
    lst["why"] = lst.student_id.map(risk_reasons)
    lst = lst[(lst.score < 45) | (lst.risk >= 30) | (lst.change < -10)].sort_values("risk", ascending=False)
    st.dataframe(lst, use_container_width=True, hide_index=True, column_config={
        "student_name": "Name", "section": "Sec", "score": st.column_config.NumberColumn("Score", format="%.0f"),
        "attendance_pct": st.column_config.NumberColumn("Attendance", format="%.0f%%"), "assignment_submission_pct": st.column_config.NumberColumn("Assignments", format="%.0f%%"),
        "change": st.column_config.NumberColumn(f"Change since {EXAMS[0]}", format="%+.0f"),
        "risk": st.column_config.ProgressColumn("Chance of failing final", min_value=0, max_value=100, format="%d%%"), "why": "Why flagged"})
    st.download_button("⬇️ Download attention list (CSV)", lst.to_csv(index=False), f"{t}_attention_list.csv")

# ================================================================== RISK
elif page == "⚠️ Students at Risk":
    st.title("Students at Risk of Failing the Final")
    if len(EXAMS) < 2:
        st.info("Needs at least two exams."); st.stop()
    rk = pd.DataFrame({"risk": risk_prob}).join(students.set_index("student_id")[["student_name", "section", "attendance_pct", "assignment_submission_pct"]])
    expl, sv, Xf = risk_explained(ss, students, attendance, KEY)
    rk["why"] = expl.raises_risk.reindex(rk.index); rk["can_change"] = expl.can_change.reindex(rk.index)
    rk = rk[rk.index.isin(VISIBLE.student_id)]
    rk["level"] = A.risk_band(rk.risk)
    c = st.columns(3)
    c[0].metric("🔴 High risk", int((rk.level == "High").sum()), help="60%+ chance of failing at least one subject in the final")
    c[1].metric("🟡 Medium risk", int((rk.level == "Medium").sum()), help="30–60% chance")
    c[2].metric("🟢 Low risk", int((rk.level == "Low").sum()))
    rec = risk_metrics.get("Recall", 0); pre = risk_metrics.get("Precision", 0)
    card(f"<b>How reliable is this?</b> In testing, the system caught <b>{rec * 100:.0f} out of 100</b> students who actually failed. "
         f"About <b>{pre * 100:.0f} of every 100</b> flagged students really failed — so use this list to <b>offer help</b>, never to label a child.", "amb")
    l, r = st.columns([2, 3])
    with l:
        imp = risk_imp.tail(7)
        show(px.bar(x=imp.values, y=imp.index, orientation="h", title="What warns us most", labels={"x": "importance", "y": ""}), 380)
    with r:
        bs = rk.groupby(["section", "level"]).size().reset_index(name="students")
        show(px.bar(bs, x="section", y="students", color="level", title="Risk by section", category_orders={"level": ["High", "Medium", "Low"]},
                    color_discrete_map={"High": "#ef5b5b", "Medium": "#f5a524", "Low": "#2fbf71"}), 380)
    lv = st.multiselect("Show", ["High", "Medium", "Low"], default=["High", "Medium"])
    out = rk[rk.level.isin(lv)].sort_values("risk", ascending=False).reset_index()
    out["risk"] = (out.risk * 100).round(0)
    st.dataframe(out, use_container_width=True, hide_index=True, column_config={
        "student_id": "ID", "student_name": "Name", "section": "Sec", "attendance_pct": st.column_config.NumberColumn("Attendance", format="%.0f%%"),
        "assignment_submission_pct": st.column_config.NumberColumn("Assignments", format="%.0f%%"),
        "risk": st.column_config.ProgressColumn("Chance of failing", min_value=0, max_value=100, format="%d%%"),
        "why": "Main reasons (points added to risk)", "can_change": "What can change quickly", "level": "Level"})
    st.download_button("⬇️ Download list (CSV)", out.to_csv(index=False), "students_at_risk.csv")
    st.subheader("Why is this student at risk?")
    pick = st.selectbox("Student", out.student_id if len(out) else rk.index, format_func=lambda x: f"{x} — {students.set_index('student_id').student_name[x]}")
    s_sv = sv.loc[pick].sort_values(key=abs, ascending=False).head(7)[::-1]
    labels_ = [f"{C3.PLAIN.get(f, (f,))[0]} = {C3.PLAIN.get(f, (f, '{}'))[1].format(Xf.loc[pick, f]) if f not in ('private_tuition', 'home_internet') else ('Yes' if Xf.loc[pick, f] else 'No')}" for f in s_sv.index]
    fig = go.Figure(go.Bar(x=s_sv.values, y=labels_, orientation="h", marker_color=["#ef5b5b" if v > 0 else "#2fbf71" for v in s_sv.values],
                           text=[f"{v:+.0f}" for v in s_sv.values]))
    fig.update_layout(title="Red pushes risk UP, green pulls it DOWN (points of risk)", xaxis_title="Effect on risk (percentage points)")
    show(fig, 340)
    card("🗣️ " + C3.student_story(pick, expl, sv, Xf, students.set_index("student_id").student_name[pick]), "amb")
    explain("For each student the system shows which facts raised or lowered their risk, and by how much. This comes from a method called SHAP. "
            "Focus on the red items that can change quickly (attendance, homework, study time, screen time).")
    with st.expander("⚖️ Is the warning system fair to every group of students?"):
        fr, fv = fairness(ss, students, attendance, KEY)
        for r_ in fv.itertuples():
            card(f"{r_.status} <b>{r_.group}</b>: {r_.verdict}", "grn" if r_.status == "🟢" else "red" if r_.status == "🔴" else "amb")
        gsel = st.selectbox("See details for", fr.group.unique())
        show(px.bar(fr[fr.group == gsel].rename(columns={"caught_pct": "Caught (of those who failed)", "false_alarm_pct": "False alarms (of those who passed)"})
                    .melt(id_vars="value", value_vars=["Caught (of those who failed)", "False alarms (of those who passed)"], var_name="measure", value_name="pct"),
                    x="value", y="pct", color="measure", barmode="group", labels={"pct": "%", "value": ""}, title=f"{gsel}: is the system equally accurate?",
                    color_discrete_map={"Caught (of those who failed)": "#2fbf71", "False alarms (of those who passed)": "#f5a524"}), 320)
        st.caption("Tested fairly: trained on earlier exams, checked on the final it had not seen. A fair system catches failing students equally well in every group.")
    with st.expander("Show technical details"):
        st.write(f"Random Forest model trained on exams up to {EXAMS[-2]} to predict {EXAMS[-1]}. AUC {risk_metrics.get('AUC', 0):.2f}, "
                 f"recall {rec:.2f}, precision {pre:.2f}, base failure rate {risk_metrics.get('base_rate', 0):.2f}.")

# ================================================================== DEEP ANALYSIS
elif page == "🔬 Deep Analysis":
    st.title("Deep Analysis — why are marks low?")
    t1, t2 = st.tabs(["What affects marks", "Types of students"])
    with t1:
        corr = A.factor_correlations(ss, students, exam)
        strength = corr.abs().mean(1).sort_values(ascending=False)
        sign = corr.mean(1)
        st.subheader("Factors ranked by effect on marks")
        for f in strength.index[:6]:
            arrow = "more → higher marks" if sign[f] > 0 else "more → lower marks"
            lvl = "Strong" if strength[f] > .3 else "Moderate" if strength[f] > .15 else "Weak"
            card(f"<b>{f}</b> — {lvl} effect ({arrow})", "grn" if sign[f] > 0 else "red")
        explain("These are patterns in the data. They show what goes together with high or low marks — they do not prove cause. Check with teachers before acting.")
        with st.expander("Show technical details (correlation table)"):
            show(px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdYlGn", zmin=-.5, zmax=.5, aspect="auto"), 460)
        f = st.selectbox("Look closer at one factor", list(A.FACTORS), format_func=A.FACTORS.get)
        d = e_now.groupby("student_id").score.mean().rename("avg").to_frame().join(students.set_index("student_id")[[f]])
        d["band"] = pd.qcut(d[f].rank(method="first"), 4, labels=["Lowest 25%", "Low-mid", "High-mid", "Highest 25%"])
        b = d.groupby("band", observed=True).avg.mean().reset_index()
        show(px.bar(b, x="band", y="avg", text=b.avg.round(1), title=f"Average score by {A.FACTORS[f]}", labels={"avg": "Average score", "band": A.FACTORS[f]}))
    with t2:
        seg, prof = A.cluster_students(ss, students, exam)
        show(px.scatter(seg, x="attendance_pct", y="avg_score", color="segment", opacity=.7, hover_data=["student_id"],
                        title="Four types of students", labels={"attendance_pct": "Attendance %", "avg_score": "Average score"}), 420)
        p = prof.set_index("segment")
        for s_ in p.sort_values("avg_score", ascending=False).index:
            base = s_.split(" (")[0]
            card(f"<b>{s_}</b> — {int(p.loc[s_, 'size'])} students · avg {p.loc[s_, 'avg_score']:.0f}, attendance {p.loc[s_, 'attendance_pct']:.0f}%<br>👉 {A.SEGMENT_ACTION.get(base, '')}",
                 "grn" if base == "Achievers" else "red" if base.startswith("Disengaged") else "amb")
        st.download_button("⬇️ Download student types (CSV)", seg.to_csv(index=False), "student_types.csv")

# ================================================================== ACTION TRACKER
elif page == "✅ Action Tracker":
    st.title("Action Tracker")
    st.caption("Record what the school did — the system checks after the next exam whether it worked.")
    acts = load_actions()
    view = acts if ROLE == "principal" else acts[(acts.subject == SCOPE) | acts.owner.str.contains(SCOPE, na=False)]
    for a in view.to_dict("records"):
        im = C3.matched_impact(ss, students, a)
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**{a['action']}**  \nProblem: {a['problem']}  \nOwner: {a['owner']} · Deadline: {a['deadline']} · Status: **{a['status']}**")
            if im:
                c1.markdown(f"### {im['verdict']}\n{im['sentence']}")
                if abs(im["naive"] - im["effect"]) > .5:
                    c1.caption(f"A simple comparison with all other students would say {im['naive']:+.1f} marks — that would be "
                               f"{'too generous' if im['naive'] > im['effect'] else 'too harsh'}, because weak students often bounce back a little on their own.")
                fig = go.Figure([go.Bar(name="Targeted students", x=["Change in marks"], y=[im["target_change"]], marker_color="#2fbf71"),
                                 go.Bar(name="Similar students (no action)", x=["Change in marks"], y=[im["matched_change"]], marker_color="#8f9bb3")])
                fig.update_layout(title="Fair comparison with similar students", barmode="group", yaxis_title="Change in marks")
                with c2: show(fig, 250)
                with c1.expander("How similar are the comparison students?"):
                    st.dataframe(im["balance"], use_container_width=True)
                    st.caption(f"Each targeted student is paired with the most similar student who did not get the action (starting marks, attendance, last year). "
                               f"Match quality: {im['match_quality']}. The range comes from re-sampling the pairs 1,000 times (bootstrap).")
            else:
                c2.info("Result will appear after the next exam is entered.")
    st.subheader("Update status")
    if len(view):
        ed = st.data_editor(view[["action_id", "action", "owner", "deadline", "status"]], hide_index=True, use_container_width=True, disabled=["action_id", "action"],
                            column_config={"status": st.column_config.SelectboxColumn("Status", options=["Not started", "In progress", "Done", "Dropped"])})
        if st.button("Save changes"):
            acts = acts.set_index("action_id"); acts.update(ed.set_index("action_id")[["owner", "deadline", "status"]])
            save_actions(acts.reset_index()); st.success("Saved."); st.rerun()
    st.subheader("Add a new action")
    with st.form("newact", clear_on_submit=True):
        c1, c2 = st.columns(2)
        prob_ = c1.text_input("Problem"); reason = c2.text_input("Why (reason)")
        action = st.text_input("Action to take")
        c1, c2, c3 = st.columns(3)
        owner = c1.text_input("Owner", value=U["display_name"]); dl = c2.date_input("Deadline", dt.date.today() + dt.timedelta(days=21))
        subj = c3.selectbox("Subject", [SCOPE] if ROLE == "hod" else ["All"] + SUBJECTS)
        tgt = st.radio("Which students?", ["Students below 50 in this subject", "High-risk students", "A section", "No specific students"], horizontal=True)
        sec = st.selectbox("Section (if 'A section')", sorted(students.section.unique()))
        if st.form_submit_button("Add action", type="primary") and action:
            last_e = ss[ss.exam_order == ss.exam_order.max()]
            ids = (last_e[(last_e.subject == subj) & (last_e.score < 50)].student_id if tgt.startswith("Students below") and subj != "All"
                   else risk_prob[risk_prob >= .6].index if tgt.startswith("High") else students[students.section == sec].student_id if tgt == "A section" else [])
            new = {"action_id": int(acts.action_id.max() + 1) if len(acts) else 1, "created": dt.date.today().isoformat(), "problem": prob_,
                   "reason": reason, "action": action, "owner": owner, "deadline": dl.isoformat(), "status": "Not started",
                   "subject": "" if subj == "All" else subj, "start_exam_order": int(ss.exam_order.max()), "target_students": ";".join(ids)}
            save_actions(pd.concat([acts, pd.DataFrame([new])], ignore_index=True)); st.success("Action added."); st.rerun()

# ================================================================== REPORT CARDS
elif page == "🎓 Report Cards":
    if ROLE == "parent":
        sid = SCOPE
    else:
        st.title("Report Cards")
        sid = st.selectbox("Student", VISIBLE.student_id, format_func=lambda x: f"{x} — {students.set_index('student_id').student_name[x]}")
    lang = st.radio("Language / భాష", ["English", "తెలుగు"], horizontal=True)
    TEL = lang != "English"; L = lambda en, te: te if TEL else en
    s = students.set_index("student_id").loc[sid]
    me = ss[ss.student_id == sid]; now = me[me.exam == exam].set_index("subject").score
    allavg = ss[ss.exam == exam].groupby("student_id").score.mean(); pct = (allavg < allavg[sid]).mean() * 100
    st.title(f"{s.student_name}"); st.caption(f"{sid} · {L('Section', R.TE['section'])} {s.section} · {exam}")
    c = st.columns(4)
    c[0].metric(L("Average", R.TE["average"]), f"{now.mean():.1f}")
    c[1].metric(L("Percentile", R.TE["percentile"]), f"{pct:.0f}", help="Scored better than this % of students in the school")
    c[2].metric(L("Attendance", R.TE["attendance"]), f"{s.attendance_pct:.0f}%")
    c[3].metric(L("Assignments", R.TE["assign"]), f"{s.assignment_submission_pct:.0f}%")
    cls = e_now[e_now.section == s.section].groupby("subject").score.mean()
    l, r = st.columns(2)
    with l:
        df = pd.DataFrame({"subject": [L(x, R.TE.get(x, x)) for x in now.index], L("My child", "మా పిల్లవాడు"): now.values, L("Class average", R.TE["class_avg"]): cls[now.index].values})
        show(px.bar(df.melt(id_vars="subject"), x="subject", y="value", color="variable", barmode="group", title=L("Marks vs class average", "మార్కులు vs తరగతి సగటు"),
                    labels={"value": L("Marks", R.TE["marks"]), "subject": ""}))
    with r:
        mm = me.sort_values("exam_order").assign(subject=lambda x: x.subject.map(lambda v: L(v, R.TE.get(v, v))))
        show(px.line(mm, x="exam", y=SC, color="subject", markers=True, title=L("Progress across exams", R.TE["progress"]), labels={"score": L("Marks", R.TE["marks"]), "exam": ""}))
    tp = marks[(marks.student_id == sid) & (marks.exam == exam)].assign(p=lambda x: x.marks / x.max_marks * 100).sort_values("p")
    card(f"<b>{L('Strengths', R.TE['strengths'])}:</b> " + ", ".join(f"{t} ({p:.0f}%)" for t, p in zip(tp.topic.tail(3)[::-1], tp.p.tail(3)[::-1])), "grn")
    card(f"<b>{L('Needs improvement', R.TE['improve'])}:</b> " + ", ".join(f"{t} – {L(sb, R.TE.get(sb, sb))} ({p:.0f}%)" for t, sb, p in zip(tp.topic.head(3), tp.subject.head(3), tp.p.head(3))), "red")
    tg = B.student_targets(ss, X["history"], sid) if X["history"] is not None else None
    if tg is not None:
        st.subheader(L("Targets for the next exam", "తదుపరి పరీక్షకు లక్ష్యాలు"))
        tg = tg.assign(Subject=tg.subject.map(lambda v: L(v, R.TE.get(v, v))))
        show(go.Figure([go.Bar(name=L("Now", "ఇప్పుడు"), x=tg.Subject, y=tg.current, marker_color="#8f9bb3"),
                        go.Bar(name=L("Target", "లక్ష్యం"), x=tg.Subject, y=tg.target_next_exam, marker_color="#2fbf71")]).update_layout(barmode="group"), 280)
        worst_t = tg.sort_values("current").iloc[0]
        card(f"🎯 <b>{s.student_name.split()[0]} {worst_t.sentence}</b>" if not TEL else
             f"🎯 <b>{s.student_name.split()[0]} తదుపరి పరీక్షలో {R.TE.get(worst_t.subject, worst_t.subject)}లో {int(worst_t.target_next_exam)} మార్కులు సాధించాలి (ఇప్పుడు {int(worst_t.current)}).</b>", "grn")
        hh = B.multi_year(X["history"], ss, sid)
        with st.expander(L("📈 Progress over the years (Class 8 → now)", "📈 సంవత్సరాల వారీగా ప్రగతి")):
            hh = hh.assign(cls=lambda d: "Class " + d["class"].astype(str), subject=lambda d: d.subject.map(lambda v: L(v, R.TE.get(v, v))))
            show(px.line(hh, x="cls", y="final_score", color="subject", markers=True, labels={"cls": "", "final_score": L("Final exam marks", "వార్షిక మార్కులు")}), 320)
    if X["question_marks"] is not None and exam == EXAMS[-1]:
        mp = B.mistake_profile(X["question_marks"], X["questions"], sid)
        with st.expander(L("🔍 What kind of mistakes? (Final exam)", "🔍 ఎలాంటి తప్పులు?")):
            st.dataframe(mp[["subject", "Recall", "Application", "Problem-solving", "diagnosis"]], hide_index=True, use_container_width=True,
                         column_config={"Recall": st.column_config.NumberColumn("Easy (recall) %", format="%d"), "Application": st.column_config.NumberColumn("Application %", format="%d"),
                                        "Problem-solving": st.column_config.NumberColumn("Hard problems %", format="%d"), "diagnosis": "What it means"})
    if ROLE != "parent":
        if st.button("✍️ Write report-card remark (English + Telugu)"):
            hist_tr = me.groupby("exam_order").score.mean()
            d, mode = ai.remarks(s.student_name, now.round(0).to_dict(), float(s.attendance_pct), float(s.assignment_submission_pct),
                                 float(hist_tr.iloc[-1] - hist_tr.iloc[0]), tg.sentence.tolist() if tg is not None else [])
            st.session_state[f"rem_{sid}"] = (d, mode)
        if st.session_state.get(f"rem_{sid}"):
            d, mode = st.session_state[f"rem_{sid}"]
            st.text_area("English remark", d["en"], key=f"ren_{sid}"); st.text_area("తెలుగు వ్యాఖ్య", d["te"], key=f"rte_{sid}")
            st.caption(f"Written by {'Claude AI' if mode == 'Claude' else 'built-in templates (add an API key for AI-written remarks)'} — please read and edit before printing.")
    st.subheader(L("Advice for parents", R.TE["advice"]))
    for kk in R.advice_keys(s, now):
        st.markdown(f"- {R.TE[kk] if TEL else R.EN_ADV[kk]}")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ " + L("Download report card (PDF, English + Telugu)", "నివేదిక డౌన్‌లోడ్ (PDF)"), R.parent_pdf(ss, students, marks, exam, sid),
                       f"report_card_{sid}.pdf", "application/pdf", type="primary")
    if ROLE in ("principal", "teacher", "hod"):
        sec = c2.selectbox("Download all report cards for section", sorted(VISIBLE.section.unique()))
        if c2.button("Prepare ZIP"):
            ids = VISIBLE[VISIBLE.section == sec].student_id
            with st.spinner(f"Creating {len(ids)} report cards…"):
                st.session_state["zip"] = (sec, R.parent_pdfs_zip(ss, students, marks, exam, ids))
        if st.session_state.get("zip"):
            z = st.session_state["zip"]; c2.download_button(f"⬇️ Section {z[0]} report cards (ZIP)", z[1], f"report_cards_section_{z[0]}.zip")

# ================================================================== PDF REPORTS
elif page == "📄 PDF Reports":
    st.title("PDF Reports")
    st.caption(f"Ready-to-print reports for {exam}. Generated automatically from the latest marks.")
    if ROLE == "principal":
        with st.container(border=True):
            st.markdown("**Principal's Exam Summary** — 2 pages: headline, goals, subject health, top 3 problems with actions, charts, action results.")
            st.download_button("⬇️ Download principal summary", R.principal_pdf(ss, students, teachers, marks, exam, targets, last_year, load_actions(), risk_prob),
                               f"principal_summary_{exam}.pdf", "application/pdf", type="primary")
    if ROLE in ("principal", "hod"):
        with st.container(border=True):
            st.markdown("**HOD subject report** — topics, teachers compared fairly, question-paper check.")
            sb = st.selectbox("Subject", [SCOPE] if ROLE == "hod" else SUBJECTS)
            st.download_button(f"⬇️ Download {sb} HOD report", R.hod_pdf(ss, students, teachers, marks, exam, sb, targets), f"hod_{sb}_{exam}.pdf", "application/pdf")
    with st.container(border=True):
        st.markdown("**Teacher sheet** — one page: weak topics, students needing attention.")
        tl = [SCOPE] if ROLE == "teacher" else teachers[teachers.teacher_id.isin(ss.teacher_id.unique()) & ((teachers.subject == SCOPE) if ROLE == "hod" else True)].teacher_id.tolist()
        tt = st.selectbox("Teacher", tl, format_func=lambda x: f"{tname[x]} — {tsubj[x]}")
        st.download_button("⬇️ Download teacher sheet", R.teacher_pdf(ss, students, teachers, marks, exam, tt, risk_prob, risk_reasons), f"teacher_{tt}_{exam}.pdf", "application/pdf")
    st.info("Parent report cards (English + Telugu) are on the **🎓 Report Cards** page, including a ZIP for a whole section.")

# ================================================================== MARKS ENTRY
elif page == "📝 Marks Entry":
    st.title("Marks Entry")
    pairs = ss[["teacher_id", "subject"]].drop_duplicates()
    if ROLE == "teacher": pairs = pairs[pairs.teacher_id == SCOPE]
    if ROLE == "hod": pairs = pairs[pairs.subject == SCOPE]
    c1, c2, c3 = st.columns(3)
    t = c1.selectbox("Teacher / class", pairs.teacher_id, format_func=lambda x: f"{tname[x]} — {tsubj[x]}")
    subj = tsubj[t]
    ex_opts = EXAMS + ["➕ New exam…"]
    ex = c2.selectbox("Exam", ex_opts, index=len(EXAMS) - 1)
    if ex == "➕ New exam…":
        ex = c3.text_input("New exam name", "Pre-Final"); ex_order = int(ss.exam_order.max()) + 1
    else:
        ex_order = int(ss[ss.exam == ex].exam_order.iloc[0])
    sids = ss[ss.teacher_id == t].student_id.unique()
    roster = students[students.student_id.isin(sids)]
    tab_x, tab_g, tab_o = st.tabs(["📊 Excel template", "⌨️ Type on screen", "📥 Import other format"])
    with tab_g:
        st.caption("Type marks directly in the table (each topic out of 25). Leave blank if absent.")
        sec_g = st.selectbox("Section", sorted(roster.section.unique()), key="g_sec")
        base = roster[roster.section == sec_g][["student_id", "student_name"]].sort_values("student_id").reset_index(drop=True)
        cur = marks[(marks.exam == ex) & (marks.subject == subj)].pivot_table(index="student_id", columns="topic", values="marks")
        for tp_ in TOPICS[subj]:
            base[tp_] = base.student_id.map(cur[tp_]) if tp_ in cur else np.nan
        ed = st.data_editor(base, hide_index=True, use_container_width=True, disabled=["student_id", "student_name"], key=f"grid_{t}_{ex}_{sec_g}",
                            column_config={tp_: st.column_config.NumberColumn(tp_, min_value=0, max_value=25, step=0.5) for tp_ in TOPICS[subj]})
        filled = ed[TOPICS[subj]].notna().sum().sum()
        st.caption(f"{filled} of {ed[TOPICS[subj]].size} cells filled")
        if st.button("💾 Save these marks", type="primary", key="g_save"):
            errs, warns, long = A.validate_marks_sheet(ed.drop(columns="student_name"), roster, TOPICS[subj])
            if errs: st.error(pd.DataFrame(errs).problem.str.cat(sep=" · "))
            else:
                save_marks(long, ex, ex_order, subj, t); st.success(f"Saved {len(ed)} students. Blank cells were saved as absent (0)."); st.rerun()
    with tab_o:
        st.caption("Use this if your marks are already in another file (school software export, government portal sheet, etc.). "
                   "Common format: one row per student, one column per subject.")
        f = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="other_up")
        if f:
            raw = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            st.dataframe(raw.head(8), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            idc = c1.selectbox("Which column has the student ID / roll number?", raw.columns)
            subj_cols = c2.multiselect("Which columns are subject marks?", [c for c in raw.columns if c != idc])
            mx = c3.number_input("Marks are out of", 1, 1000, 100)
            idmap = None
            if not raw[idc].astype(str).str.upper().isin(students.student_id).any():
                st.warning("These IDs don't match our student IDs. Upload a mapping file (two columns: their_id, student_id), or use the Pilot Toolkit to create anonymous IDs.")
                mf = st.file_uploader("ID mapping file", type=["csv", "xlsx"], key="idmap")
                if mf: idmap = (pd.read_csv(mf) if mf.name.endswith(".csv") else pd.read_excel(mf)).astype(str)
            if subj_cols:
                long_o = B.wide_to_long(raw.astype({idc: str}), idc, subj_cols, mx)
                if idmap is not None: long_o["student_id"] = long_o.student_id.map(dict(zip(idmap.iloc[:, 0], idmap.iloc[:, 1])))
                long_o["student_id"] = long_o.student_id.astype(str).str.upper()
                smap = {c: st.selectbox(f"'{c}' is which subject?", SUBJECTS, index=SUBJECTS.index(ai._find(SUBJECTS, c)) if ai._find(SUBJECTS, c) else 0, key=f"sm_{c}") for c in subj_cols}
                long_o["subject"] = long_o.subject.map(smap)
                bad = long_o[~long_o.student_id.isin(students.student_id) | (long_o.score > 100) | (long_o.score < 0)]
                n_abs = int(long_o.score.isna().sum())
                if n_abs: st.caption(f"{n_abs} blank / AB entries will be saved as absent (not counted in averages).")
                st.write(f"✅ {len(long_o) - len(bad)} marks ready · ❌ {len(bad)} problems")
                if len(bad): st.dataframe(bad.head(20), hide_index=True)
                if st.button("💾 Import", type="primary", disabled=len(bad) > 0, key="imp_go"):
                    # spread the subject total equally over that subject's topics so all pages keep working
                    rows = []
                    tmap_ = ss[ss.exam_order == ss.exam_order.max()][["student_id", "subject", "teacher_id"]]
                    for r_ in long_o.merge(tmap_, on=["student_id", "subject"], how="left").itertuples():
                        for tp_ in TOPICS[r_.subject]:
                            rows.append({"student_id": r_.student_id, "exam": ex, "exam_order": ex_order, "subject": r_.subject, "topic": tp_,
                                         "teacher_id": r_.teacher_id, "marks": 0 if pd.isna(r_.score) else round(r_.score / 100 * 25, 1), "max_marks": 25,
                                         "absent": int(pd.isna(r_.score))})
                    new = pd.DataFrame(rows)
                    keep = ~((marks.exam == ex) & marks.set_index(["student_id", "subject"]).index.isin(new.set_index(["student_id", "subject"]).index))
                    pd.concat([marks[keep], new[marks.columns]], ignore_index=True).to_csv(P("marks.csv"), index=False); st.cache_data.clear()
                    st.success(f"Imported {len(long_o)} subject marks for {ex}. (Topic-level detail is not available from this format.)")
    tab_x.markdown("**Step 1** Download the Excel template → **Step 2** type marks in the yellow cells → **Step 3** upload it here. The system checks for mistakes before saving.")
    tab_x.download_button("⬇️ Step 1 — Download Excel template", R.marks_template(roster, subj, TOPICS[subj], ex, tname[t]),
                       f"marks_{subj}_{ex}_{t}.xlsx".replace(" ", "_"), type="primary")
    with tab_x:
        up = st.file_uploader("Step 3 — Upload the filled template", type=["xlsx", "csv"])
        if up:
            df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up, sheet_name="Marks")
            errs, warns, long = A.validate_marks_sheet(df, roster, TOPICS[subj])
            if errs:
                st.error(f"❌ Found {len(errs)} mistake(s). Please fix them in Excel and upload again. Nothing was saved.")
                st.dataframe(pd.DataFrame(errs), hide_index=True, use_container_width=True, column_config={"row": "Excel row", "problem": "Problem"})
            else:
                st.success(f"✅ No mistakes. {long.student_id.nunique()} students × {len(TOPICS[subj])} topics ready to save.")
                if warns:
                    with st.expander(f"⚠️ {len(warns)} note(s) — please check"):
                        st.dataframe(pd.DataFrame(warns), hide_index=True, use_container_width=True)
                st.dataframe(long.pivot_table(index="student_id", columns="topic", values="marks").head(10), use_container_width=True)
                if st.button("💾 Save marks", type="primary"):
                    save_marks(long, ex, ex_order, subj, t)
                    st.success("Saved. All dashboards and the risk model are updated."); st.balloons()
    if ROLE == "principal":
        with st.expander("Admin: replace whole data files (CSV)"):
            for n in ["students", "teachers", "marks", "attendance", "last_year"]:
                f = st.file_uploader(f"{n}.csv", type=["csv"], key=f"adm_{n}")
                if f is not None and st.button(f"Replace {n}.csv", key=f"rep_{n}"):
                    pd.read_csv(f).to_csv(P(f"{n}.csv"), index=False); st.cache_data.clear(); st.success(f"{n}.csv replaced.")

# ================================================================== EARLY WARNINGS
elif page == "🔔 Early Warnings":
    st.title("Early Warnings")
    st.caption("Signs of trouble found today — without waiting for exams.")
    if X["daily_attendance"] is None:
        st.info("Daily attendance data not available."); st.stop()
    ew = B.early_warnings(X["daily_attendance"], students, sections, hw_per)
    if ROLE == "teacher":
        my_secs = sections[sections.class_teacher_id == SCOPE].section.tolist()
        ew = ew[ew.section.isin(my_secs) | (ew.student_id.isin(VISIBLE.student_id) & ew.alert.str.contains(tsubj[SCOPE]))]
        if my_secs: st.caption(f"You are class teacher of Section {', '.join(my_secs)} — showing attendance alerts for it and homework alerts for your subject.")
    elif ROLE == "hod":
        ew = ew[~ew.alert.str.startswith("Missed") | ew.alert.str.contains(SCOPE)]
    last_day = X["daily_attendance"].date.max()
    c = st.columns(3)
    c[0].metric("🔴 Absent 3+ days in a row", int(ew.alert.str.contains("in a row").sum()))
    c[1].metric("🟡 Low attendance (30 days)", int(ew.alert.str.contains("last 30").sum()))
    c[2].metric("📘 Homework missed 3 times", int(ew.alert.str.startswith("Missed").sum()))
    st.caption(f"Based on attendance up to {last_day}.")
    ew["class_teacher"] = ew.class_teacher_id.map(tname)
    order = {"High": 0, "Medium": 1}
    st.dataframe(ew.sort_values("level", key=lambda x: x.map(order)).drop(columns="class_teacher_id"), hide_index=True, use_container_width=True,
                 column_config={"student_name": "Name", "section": "Sec", "alert": "Warning", "level": "Level", "action": "What to do", "class_teacher": "Class teacher"})
    st.download_button("⬇️ Download today's call list (CSV)", ew.to_csv(index=False), f"early_warnings_{last_day}.csv")
    explain("A student absent 3 or more school days in a row often has a problem at home, health or motivation. Calling the parent the same day works far better than waiting for exam results.")

# ================================================================== DAILY ATTENDANCE
elif page == "📅 Attendance":
    st.title("Daily Attendance")
    da = X["daily_attendance"]
    my_secs = sorted(VISIBLE.section.unique())
    c1, c2 = st.columns(2)
    sec_a = c1.selectbox("Section", my_secs)
    day = c2.date_input("Date", dt.date.fromisoformat(da.date.max()) if da is not None else dt.date.today())
    roster = students[students.section == sec_a][["student_id", "student_name"]].sort_values("student_id").reset_index(drop=True)
    prev = da[(da.date == day.isoformat()) & da.student_id.isin(roster.student_id)].set_index("student_id").present if da is not None else pd.Series(dtype=int)
    roster["Absent"] = roster.student_id.map(lambda x: prev.get(x, 1) == 0)
    st.caption("Tick only the students who are **absent**. Everyone else is marked present. Works on a phone.")
    ed = st.data_editor(roster, hide_index=True, use_container_width=True, disabled=["student_id", "student_name"], key=f"att_{sec_a}_{day}",
                        column_config={"Absent": st.column_config.CheckboxColumn("Absent ✗", width="small"), "student_name": "Name", "student_id": "ID"}, height=480)
    n_abs = int(ed.Absent.sum())
    st.markdown(f"**{len(ed) - n_abs} present · {n_abs} absent**")
    if st.button("💾 Save attendance", type="primary", use_container_width=True):
        new = pd.DataFrame({"student_id": ed.student_id, "date": day.isoformat(), "present": (~ed.Absent).astype(int)})
        base = da[~((da.date == day.isoformat()) & da.student_id.isin(new.student_id))] if da is not None else pd.DataFrame(columns=new.columns)
        pd.concat([base, new], ignore_index=True).sort_values(["date", "student_id"]).to_csv(P("daily_attendance.csv"), index=False)
        st.cache_data.clear(); st.success(f"Saved attendance for Section {sec_a} on {day}."); st.rerun()
    if da is not None:
        rec = da[da.student_id.isin(roster.student_id)].groupby("date").present.mean().mul(100).tail(30).reset_index()
        show(px.bar(rec, x="date", y="present", title=f"Section {sec_a}: attendance % — last 30 school days", labels={"present": "% present", "date": ""}), 280)

# ================================================================== HOMEWORK
elif page == "📘 Homework":
    st.title("Homework")
    hw, hws = X["homework"], X["homework_status"]
    subj_h = tsubj[SCOPE] if ROLE == "teacher" else SCOPE if ROLE == "hod" else st.selectbox("Subject", SUBJECTS)
    t1, t2 = st.tabs(["📊 Does homework matter?", "✏️ Mark homework"])
    with t1:
        per = hw_per[hw_per.subject == subj_h]
        per = per[per.student_id.isin(VISIBLE.student_id)]
        bands, r_ = B.homework_effect(per, ss, exam)
        c = st.columns(3)
        c[0].metric("Average submission", f"{per.submitted_pct.mean():.0f}%")
        c[1].metric("Students missing 3 in a row", int((per.missing_last3 >= 3).sum()))
        c[2].metric("Link with marks", "Strong" if r_ > .3 else "Moderate" if r_ > .15 else "Weak", help=f"correlation {r_:.2f}")
        show(px.bar(bands, x="band", y="mean", text=bands["mean"].round(1), title=f"{subj_h}: marks by homework submission",
                    labels={"band": "Homework submitted", "mean": "Average marks"}, color="mean", color_continuous_scale="RdYlGn").update_layout(coloraxis_showscale=False), 320)
        explain("Students who submit more homework score higher. Following up on missing homework early is one of the cheapest ways to raise marks.")
        worst = per.sort_values("submitted_pct").merge(students[["student_id", "student_name", "section"]], on="student_id").head(20)
        st.subheader("Lowest homework submission")
        st.dataframe(worst[["student_id", "student_name", "section", "submitted_pct", "missing", "missing_last3"]], hide_index=True, use_container_width=True,
                     column_config={"submitted_pct": st.column_config.ProgressColumn("Submitted", min_value=0, max_value=100, format="%d%%"),
                                    "missing": "Missed (total)", "missing_last3": "Missed (last 3)"})
    with t2:
        mine_hw = hw[hw.subject == subj_h].sort_values("date", ascending=False)
        c1, c2 = st.columns([2, 1])
        with c2.form("newhw", clear_on_submit=True):
            st.markdown("**New homework**")
            ttl = st.text_input("Title", f"{subj_h} worksheet"); dd = st.date_input("Date given", dt.date.today())
            if st.form_submit_button("Create"):
                nid = f"H{int(hw.hw_id.str[1:].astype(int).max()) + 1:03d}"
                pd.concat([hw, pd.DataFrame([{"hw_id": nid, "subject": subj_h, "date": dd.isoformat(), "title": ttl}])]).to_csv(P("homework.csv"), index=False)
                st.cache_data.clear(); st.rerun()
        hid = c1.selectbox("Homework", mine_hw.hw_id, format_func=lambda x: f"{mine_hw.set_index('hw_id').title[x]} — {mine_hw.set_index('hw_id').date[x]}")
        sec_h = c1.selectbox("Section", sorted(VISIBLE.section.unique()), key="hw_sec")
        rost = students[(students.section == sec_h) & students.student_id.isin(VISIBLE.student_id)][["student_id", "student_name"]]
        cur = hws[hws.hw_id == hid].set_index("student_id").status
        rost = rost.assign(status=rost.student_id.map(cur).fillna("Missing"))
        ed = st.data_editor(rost, hide_index=True, use_container_width=True, disabled=["student_id", "student_name"], key=f"hwed_{hid}_{sec_h}",
                            column_config={"status": st.column_config.SelectboxColumn("Status", options=["Submitted", "Late", "Missing"])})
        if st.button("💾 Save homework status", type="primary"):
            keep = ~((hws.hw_id == hid) & hws.student_id.isin(ed.student_id))
            pd.concat([hws[keep], ed[["student_id", "status"]].assign(hw_id=hid)[hws.columns]]).to_csv(P("homework_status.csv"), index=False)
            st.cache_data.clear(); st.success("Saved."); st.rerun()

# ================================================================== TARGETS & PROGRESS
elif page == "🎯 Targets & Progress":
    st.title("Targets & Progress")
    tg = targets_table(ss, X["history"], KEY + XKEY)
    tg = tg[tg.student_id.isin(VISIBLE.student_id)]
    if ROLE == "teacher": tg = tg[tg.subject == tsubj[SCOPE]]
    if ROLE == "hod": tg = tg[tg.subject == SCOPE]
    t1, t2 = st.tabs(["🎯 Next-exam targets", "📈 Year-on-year (Class 8 → now)"])
    with t1:
        st.caption("Each student gets a realistic target from their **own** marks this year and in past years — not one target for everyone.")
        tg = tg.merge(students[["student_id", "student_name", "section"]], on="student_id")
        tg["gap"] = tg.target_next_exam - tg.current
        f_sec = st.multiselect("Section", sorted(tg.section.unique()), default=sorted(tg.section.unique()))
        v = tg[tg.section.isin(f_sec)]
        c = st.columns(3)
        c[0].metric("Students below pass mark now", int((v.current < A.PASS).sum()))
        c[1].metric("Average target gain", f"+{v.gap.mean():.1f} marks")
        c[2].metric("If every target is met, failures →", int((v.target_next_exam < A.PASS).sum()))
        st.dataframe(v[["student_id", "student_name", "section", "subject", "current", "best_ever", "trend_per_exam", "target_next_exam", "sentence"]].sort_values("current"),
                     hide_index=True, use_container_width=True,
                     column_config={"current": "Now", "best_ever": "Best ever", "trend_per_exam": st.column_config.NumberColumn("Trend / exam", format="%+.1f"),
                                    "target_next_exam": "🎯 Target", "sentence": "In words"})
        st.download_button("⬇️ Download targets (CSV)", v.to_csv(index=False), "student_targets.csv")
    with t2:
        allh = B.multi_year(X["history"], ss)
        allh = allh[allh.student_id.isin(VISIBLE.student_id)]
        if ROLE == "teacher": allh = allh[allh.subject == tsubj[SCOPE]]
        if ROLE == "hod": allh = allh[allh.subject == SCOPE]
        tr = allh.groupby(["class", "subject"]).final_score.mean().reset_index()
        tr["class"] = "Class " + tr["class"].astype(str)
        show(px.line(tr, x="class", y="final_score", color="subject", markers=True, title="Same students, followed over the years", labels={"final_score": "Average final marks", "class": ""}), 360)
        mv = B.cohort_movement(allh)
        show(px.bar(mv.melt(id_vars="subject", var_name="movement", value_name="students"), x="subject", y="students", color="movement",
                    color_discrete_map={"Improved 5+": "#2fbf71", "Stable": "#8f9bb3", "Dropped 5+": "#ef5b5b"}, title="Last year → this year: who improved, who dropped"), 340)
        explain("This follows the same students from Class 8. A subject where many students drop when they move up usually means the jump in syllabus difficulty is not being supported.")

# ================================================================== MISTAKE ANALYSIS
elif page == "🔍 Mistake Analysis":
    st.title("Mistake Analysis — why are marks lost?")
    if X["question_marks"] is None:
        st.info("Needs question-wise marks."); st.stop()
    st.caption(f"Based on question-wise marks in the {EXAMS[-1]}. Questions are tagged Easy (recall), Application, and Hard (problem-solving).")
    mp = mistakes(X["question_marks"], X["questions"], XKEY)
    mp = mp[mp.student_id.isin(VISIBLE.student_id)]
    subj_m = tsubj[SCOPE] if ROLE == "teacher" else SCOPE if ROLE == "hod" else st.selectbox("Subject", SUBJECTS)
    m1 = mp[mp.subject == subj_m].merge(students[["student_id", "student_name", "section"]], on="student_id")
    cnt = m1.diagnosis.value_counts().reset_index()
    l, r = st.columns(2)
    with l:
        show(px.pie(cnt, names="diagnosis", values="count", hole=.5, title=f"{subj_m}: type of problem",
                    color="diagnosis", color_discrete_map={"Careless mistakes": "#f5a524", "Concept gap": "#ef5b5b", "Needs hard-problem practice": "#4f8ff7",
                                                           "Strong": "#2fbf71", "Steady — practise more": "#8f9bb3"}), 360)
    with r:
        qst = B.question_stats(X["question_marks"][X["question_marks"].student_id.isin(VISIBLE.student_id)], X["questions"])
        qst = qst[qst.subject == subj_m]
        show(px.bar(qst.groupby(["topic", "type"]).avg_pct.mean().reset_index(), x="topic", y="avg_pct", color="type", barmode="group",
                    title="% scored by question type", labels={"avg_pct": "% scored"},
                    color_discrete_map={"Recall": "#2fbf71", "Application": "#f5a524", "Problem-solving": "#ef5b5b"}), 360)
    for d_, n_ in cnt.itertuples(index=False):
        card(f"<b>{d_}</b> — {n_} students. 👉 {B.MISTAKE_ACTION[d_]}", {"Careless mistakes": "amb", "Concept gap": "red", "Strong": "grn"}.get(d_, ""))
    explain("If a student gets hard questions right but easy ones wrong, the problem is carelessness, not understanding. If easy questions are also wrong, the basics are missing. "
            "Each needs a different fix.")
    st.subheader("Students")
    pick = st.multiselect("Show", cnt.diagnosis.tolist(), default=[d for d in ["Concept gap", "Careless mistakes"] if d in cnt.diagnosis.tolist()])
    st.dataframe(m1[m1.diagnosis.isin(pick)][["student_id", "student_name", "section", "Recall", "Application", "Problem-solving", "diagnosis"]], hide_index=True, use_container_width=True)
    with st.expander("Hardest individual questions"):
        st.dataframe(qst.sort_values("avg_pct").head(10), hide_index=True, use_container_width=True)

# ================================================================== WORKLOAD
elif page == "⚖️ Teacher Workload":
    st.title("Teacher Workload")
    st.caption("Low results are sometimes caused by too much work, not poor teaching. Check this before judging a teacher.")
    w = B.teacher_workload(teachers, ss, students, exam)
    if ROLE == "hod": w = w[w.subject == SCOPE]
    c = st.columns(3)
    c[0].metric("Overloaded (34+ periods/week)", int((w.load == "Overloaded").sum()))
    c[1].metric("Average periods / week", f"{w.periods_per_week.mean():.0f}")
    c[2].metric("Largest class", f"{int(w.avg_class_size.max())} students")
    show(px.scatter(w, x="periods_per_week", y="value_added", size="students", color="load", hover_name="teacher_name", text="teacher_name",
                    color_discrete_map={"Normal": "#2fbf71", "Heavy": "#f5a524", "Overloaded": "#ef5b5b"},
                    labels={"periods_per_week": "Periods per week", "value_added": "Marks above (+) / below (−) expected"},
                    title="Workload vs results").update_traces(textposition="top center", textfont_size=9), 460)
    over = w[(w.load == "Overloaded") & (w.value_added < -2)]
    for r_ in over.itertuples():
        card(f"<b>{r_.teacher_name}</b> ({r_.subject}) teaches {r_.periods_per_week} periods/week plus {r_.other_duties}, and students score {abs(r_.value_added):.1f} marks below expected. "
             f"👉 Reduce other duties before judging teaching quality.", "red")
    st.dataframe(w[["teacher_name", "subject", "sections", "classes", "students", "avg_class_size", "periods_per_week", "other_duties", "experience_yrs", "load", "value_added"]].round(1),
                 hide_index=True, use_container_width=True, column_config={"value_added": st.column_config.NumberColumn("vs expected", format="%+.1f")})

# ================================================================== AI ASSISTANT
elif page == "🤖 AI Assistant":
    st.title("AI Assistant")
    if ai.llm_available():
        st.success("Connected to Claude AI.", icon="✅")
    else:
        st.info("Running in **offline mode** (built-in rules and templates). To use Claude AI, set `ANTHROPIC_API_KEY` — see README.", icon="ℹ️")
    tabs = st.tabs(["💬 Ask the data", "🗒️ Teacher feedback", "📄 Remedial worksheet"])
    with tabs[0]:
        st.caption("Ask in plain English. Answers are calculated from your school's data.")
        extra = dict(daily=X["daily_attendance"], sections=sections, hw_per=hw_per, qm=X["question_marks"], qs=X["questions"], history=X["history"],
                     explain=lambda: risk_explained(ss, students, attendance, KEY))
        tools = ai.Tools(ss, students, teachers, marks, exam, risk_prob, allowed_students=None if ROLE in ("principal", "hod") else VISIBLE.student_id, extra=extra)
        if "chat" not in st.session_state: st.session_state.chat = []
        ex_q = ["Which section is weakest in Chemistry and why?", "Why are Maths marks low?", "Who is at risk in Section B?", f"Why is {risk_prob.reindex(VISIBLE.student_id).idxmax()} at risk and what should we do?"]
        cols = st.columns(len(ex_q)); clicked = None
        for c_, q_ in zip(cols, ex_q):
            if c_.button(q_, use_container_width=True): clicked = q_
        for m_ in st.session_state.chat:
            with st.chat_message(m_["role"]):
                st.markdown(m_["content"])
                for ttl, df in m_.get("tables", []):
                    if isinstance(df, pd.DataFrame) and len(df) > 1 and df.shape[1] >= 2:
                        num = df.select_dtypes("number").columns
                        if len(num):
                            show(px.bar(df, x=df.columns[0], y=num[0], title=ttl), 260)
        q = st.chat_input("e.g. Which section is weakest in Chemistry and why?") or clicked
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            with st.spinner("Analysing…"):
                hist = [{"role": m_["role"], "content": m_["content"]} for m_ in st.session_state.chat[:-1]]
                ans, tabs_, mode = ai.ask_data(q, tools, hist)
            st.session_state.chat.append({"role": "assistant", "content": ans + ("\n\n*— answered by Claude AI*" if mode == "Claude" else "\n\n*— offline mode*"), "tables": tabs_[:2]})
            st.rerun()
        if st.session_state.chat and st.button("Clear chat"): st.session_state.chat = []; st.rerun()
    with tabs[1]:
        tl = [SCOPE] if ROLE == "teacher" else teachers[teachers.teacher_id.isin(ss.teacher_id.unique()) & ((teachers.subject == SCOPE) if ROLE == "hod" else True)].teacher_id.tolist()
        tt = st.selectbox("Teacher", tl, format_func=lambda x: f"{tname[x]} — {tsubj[x]}", key="fb_t")
        sj = tsubj[tt]; mine = e_now[e_now.teacher_id == tt]
        tm = marks[(marks.exam == exam) & (marks.teacher_id == tt)].groupby("topic").marks.mean().sort_values()
        va = A.value_added(ss, students, teachers, exam).set_index("teacher_id").value_added
        stats = dict(class_avg=round(mine.score.mean(), 1), school_avg=round(e_now[e_now.subject == sj].score.mean(), 1), value_added=round(va.get(tt, 0), 1),
                     weak_topics=tm.index[:2].tolist(), strong_topic=tm.index[-1], fails=int((mine.score < A.PASS).sum()),
                     periods=int(teachers.set_index("teacher_id").get("periods_per_week", pd.Series(dtype=int)).get(tt, 0)))
        if st.button("✍️ Write feedback", type="primary"):
            st.session_state["fb"] = ai.teacher_feedback(tname[tt], sj, stats)
        if st.session_state.get("fb"):
            txt, mode = st.session_state["fb"]
            st.text_area("Feedback (edit before sending)", txt, height=280)
            st.caption("Written by " + ("Claude AI" if mode == "Claude" else "built-in template"))
    with tabs[2]:
        c1, c2, c3, c4 = st.columns(4)
        sj = c1.selectbox("Subject", [tsubj[SCOPE]] if ROLE == "teacher" else [SCOPE] if ROLE == "hod" else SUBJECTS, key="ws_s")
        weak = (marks[(marks.subject == sj) & (marks.exam == exam) & marks.student_id.isin(VISIBLE.student_id)].groupby("topic").marks.mean()).sort_values()
        tp_ = c2.selectbox("Topic (weakest first)", weak.index.tolist(), format_func=lambda x: f"{x} ({weak[x] / 25 * 100:.0f}% scored)")
        lv = c3.selectbox("Level", ["Easy", "Medium", "Hard"], index=1)
        n_ = c4.number_input("Questions", 5, 20, 10)
        if st.button("📄 Create worksheet", type="primary"):
            st.session_state["ws"] = ai.worksheet(sj, tp_, lv, int(n_), seed=np.random.randint(1e6))
        if st.session_state.get("ws"):
            md, mode = st.session_state["ws"]
            st.markdown(md)
            st.download_button("⬇️ Download worksheet (.md)", md, f"worksheet_{sj}_{tp_}.md")
            st.caption("Made by " + ("Claude AI" if mode == "Claude" else "built-in question generator") + " — please check before giving to students.")

# ================================================================== AI AGENTS
elif page == "🧠 AI Agents":
    st.title("AI Agents")
    st.caption("Agents watch the data, decide what matters and prepare the work. **They never send a message or change marks — a person approves everything.**")

    @st.cache_resource
    def agent_data(_k, _day):
        return AG.SchoolData(DATA)
    d = agent_data((KEY, XKEY, mtime("actions.csv", "agent_queue.csv")), dt.date.today())
    CLAUDE = ai.llm_available()
    (st.success if CLAUDE else st.info)("Claude AI is connected — agents use it for writing and planning." if CLAUDE else
                                         "Offline mode: agents use built-in rules and templates. Add `ANTHROPIC_API_KEY` to use Claude (see README).", icon="✅" if CLAUDE else "ℹ️")
    ICON = {"parent_alert": "📞", "action_proposal": "✅", "data_check": "🧪", "remedial_plan": "🩹"}
    who = lambda o: f"{d.tname[o]} ({o})" if o in d.tname else o
    owner_f = SCOPE if ROLE == "teacher" else None

    def inbox(kinds, prefix):
        q = AG.visible_queue(d, ROLE, SCOPE, "Pending", kinds)
        if not len(q):
            st.success("Nothing is waiting for approval here.", icon="✅"); return
        q = q.assign(_o=q.level.map({"High": 0, "Medium": 1}).fillna(2)).sort_values(["_o", "item_id"])
        st.caption(f"{len(q)} waiting for a person to approve. Nothing is sent or changed until you press Approve.")
        for r in q.head(25).itertuples():
            with st.container(border=True):
                lv = {"High": "🔴", "Medium": "🟡"}.get(r.level, "⚪")
                st.markdown(f"**{ICON.get(r.kind, '•')} {r.title}**  \n<span style='color:#9aa6bd;font-size:.85rem'>{lv} {r.level} · {r.agent} · for {who(r.owner)} · {r.item_id}</span>", unsafe_allow_html=True)
                en, te = r.body_en, r.body_te
                if r.kind == "parent_alert":
                    st.info("📞 **Call note for the class teacher:** " + r.body_note)
                    c1, c2 = st.columns(2)
                    en = c1.text_area("Message to parent — English (edit if needed)", r.body_en, height=170, key=f"{prefix}en{r.item_id}")
                    te = c2.text_area("Message to parent — Telugu (edit if needed)", r.body_te, height=170, key=f"{prefix}te{r.item_id}")
                else:
                    st.markdown(r.body_en)
                    if r.kind == "remedial_plan":
                        pl = json.loads(r.payload)
                        for wk, md in pl["worksheets"].items():
                            with st.expander(f"📄 {wk} worksheet"): st.markdown(md)
                b1, b2, _ = st.columns([1.3, 1, 5])
                if b1.button("✅ Approve", key=f"{prefix}ok{r.item_id}", type="primary"):
                    msg = AG.decide(d, r.item_id, "Approved", U["display_name"], en if r.kind == "parent_alert" else None, te if r.kind == "parent_alert" else None)
                    st.toast(msg); st.cache_data.clear(); st.rerun()
                if b2.button("✖ Reject", key=f"{prefix}no{r.item_id}"):
                    AG.decide(d, r.item_id, "Rejected", U["display_name"]); st.rerun()

    TABS = {"principal": ["📰 Weekly briefing", "✅ Action review", "🔔 Early-warning drafts", "🩹 Remedial planner", "🧪 Data checks", "📥 Approval inbox", "🧾 Agent log"],
            "hod": ["✅ Action review", "🩹 Remedial planner", "🧪 Data checks", "📥 Approval inbox"],
            "teacher": ["🔔 Early-warning drafts", "🩹 Remedial planner", "🧪 Data checks", "📥 Approval inbox"]}[ROLE]
    npend = len(AG.visible_queue(d, ROLE, SCOPE))
    if ROLE == "principal":
        c1, c2 = st.columns([1, 4])
        if c1.button("▶ Run all agents now", type="primary", help="Same as the scheduled run: briefing, action review, early warnings, remedial plans, data checks."):
            with st.spinner("Agents are working…"):
                out = AG.run_all(d, use_llm=True)
                st.session_state["brief"] = out["briefing"]
            st.cache_data.clear(); st.rerun()
        c2.markdown(f"<div style='padding-top:.5rem'>📥 <b>{npend}</b> item(s) waiting for approval</div>", unsafe_allow_html=True)

    T = dict(zip(TABS, st.tabs(TABS)))

    # ---- 1. weekly briefing
    if "📰 Weekly briefing" in T:
        with T["📰 Weekly briefing"]:
            st.caption("Every Monday the agent checks attendance, early warnings, homework, risk, actions and data quality, then picks the 3 things to act on.")
            if st.button("▶ Prepare this week's briefing"):
                with st.spinner("Gathering facts…"):
                    st.session_state["brief"] = AG.weekly_briefing(d)
                os.makedirs(d.path("agent_outputs"), exist_ok=True)
                open(d.path(f"agent_outputs/briefing_{d.today}.md"), "w").write(AG.briefing_markdown(st.session_state["brief"]))
            b = st.session_state.get("brief")
            if b is None:
                saved = sorted(f for f in os.listdir(d.path("agent_outputs"))) if os.path.isdir(d.path("agent_outputs")) else []
                saved = [f for f in saved if f.startswith("briefing_") and f.endswith(".json")]
                if saved: b = json.load(open(d.path(f"agent_outputs/{saved[-1]}")))
            if b is None:
                st.info("No briefing yet — press the button above.")
            else:
                st.markdown(f'<div class="headline">📌 {b["headline"]}</div>', unsafe_allow_html=True)
                cols = st.columns(len(b["tiles"]))
                for c_, (k_, v_) in zip(cols, b["tiles"].items()): c_.metric(k_, v_)
                for i, x in enumerate(b["items"]):
                    card(f"""<div style="font-size:1.05rem;font-weight:700;margin-bottom:6px">{i + 1}. {x['problem']}</div><div class="pp">
                    <b>Why</b><span>{x['why']}</span><b>Action</b><span>{x['action']}</span>
                    <b>Owner</b><span>{x['owner']}</span><b>Deadline</b><span>{x['deadline']}</span></div>""", "red" if i == 0 else "amb")
                for g_ in b.get("good_news", []): card("🎉 " + g_, "grn")
                if b.get("watch"): st.markdown("**Also watching:** " + " · ".join(w_["problem"] for w_ in b["watch"]))
                if b.get("note"): st.warning(b["note"])
                st.caption(f"Prepared for {b['date']} · {b['exam']} · written by {'Claude AI' if b['mode'] == 'Claude' else 'built-in rules'}")
                if b.get("tool_calls"):
                    with st.expander("🔎 What the agent looked at (tool calls)"):
                        st.dataframe(pd.DataFrame(b["tool_calls"]), hide_index=True, use_container_width=True)
                st.download_button("⬇️ Download briefing (.md)", AG.briefing_markdown(b), f"briefing_{b['date']}.md")

    # ---- 2. action review
    if "✅ Action review" in T:
        with T["✅ Action review"]:
            st.caption("After each exam the agent measures every open action against similar students, then recommends **close**, **extend** or **change approach**.")
            res = AG.review_actions(d, queue=False)
            recs = [r for r in res["recs"] if ROLE == "principal" or r["subject"] == SCOPE or SCOPE in r["owner"]]
            if not recs: st.info("No open actions to review.")
            COL = {"Close": "grn", "Extend": "amb", "Change": "red"}
            for r in recs:
                im = r["impact"]
                with st.container(border=True):
                    c1, c2 = st.columns([3, 2])
                    c1.markdown(f"**#{r['action_id']} · {r['action']}**  \nOwner: {r['owner']} · Deadline: {r['deadline']}{' · ⚠️ **overdue**' if r['overdue'] else ''} · Status: {r['status']}")
                    c1.markdown(f'<div class="find {COL.get(r["decision"].split()[0], "")}"><b>Agent recommends: {r["decision"]}</b><br>{r["next_step"]}</div>', unsafe_allow_html=True)
                    if im:
                        fig = go.Figure(go.Scatter(x=[im["effect"]], y=["Effect of the action"], mode="markers", marker=dict(size=14, color="#2fbf71" if im["worked"] else "#f5a524"),
                                                   error_x=dict(type="data", symmetric=False, array=[im["hi"] - im["effect"]], arrayminus=[im["effect"] - im["lo"]], color="#9aa6bd")))
                        fig.add_vline(x=0, line_dash="dash", line_color="#9aa6bd")
                        fig.update_layout(title="Gain vs similar students", xaxis_title="marks", showlegend=False)
                        with c2: show(fig, 190)
            if recs and st.button("📥 Send these recommendations to the approval inbox", type="primary"):
                n = AG.review_actions(d, owner_filter=None if ROLE == "principal" else SCOPE)["items_created"]
                st.success(f"{n} proposal(s) added to the Approval inbox."); st.rerun()
            with st.expander("Waiting for approval"): inbox(["action_proposal"], "ar")

    # ---- 3. early-warning drafts
    if "🔔 Early-warning drafts" in T:
        with T["🔔 Early-warning drafts"]:
            st.caption("Students absent 3+ days in a row (or under 65% attendance in 30 days): the agent drafts a **parent message in English and Telugu** and a **call note for the class teacher**.")
            if st.button("▶ Run early-warning agent", type="primary"):
                with st.spinner("Checking attendance…"):
                    r_ = AG.early_warning_agent(d, owner_filter=owner_f)
                st.success(f"{len(r_['flagged'])} students flagged · {r_['items_created']} new drafts added · written by {'Claude AI' if r_['mode'] == 'Claude' else 'built-in templates'}"); st.session_state["ew"] = r_["flagged"]
            fl = st.session_state.get("ew")
            if fl is not None and len(fl):
                with st.expander(f"📋 Call list — {len(fl)} students", expanded=False):
                    st.dataframe(fl.sort_values("level"), hide_index=True, use_container_width=True)
            inbox(["parent_alert"], "ew")

    # ---- 4. remedial planner
    if "🩹 Remedial planner" in T:
        with T["🩹 Remedial planner"]:
            st.caption("Diagnoses the kind of mistakes (careless / concept gap / hard problems), builds a **2-week plan with worksheets**, then **re-checks after the next exam**.")
            mode_ = st.radio("Who is it for?", ["One student", "Let the agent pick the highest-risk students"], horizontal=True)
            if mode_ == "One student":
                c1, c2 = st.columns(2)
                sid_ = c1.selectbox("Student", VISIBLE.student_id, format_func=lambda x: f"{x} — {d.sname[x]}", key="rp_s")
                subs_ = ["(weakest subject)"] + SUBJECTS
                sj_ = c2.selectbox("Subject", subs_, key="rp_sub")
                if st.button("🩹 Build plan", type="primary"):
                    st.session_state["rp"] = AG.remedial_plan(d, sid_, None if sj_ == subs_[0] else sj_)
                p_ = st.session_state.get("rp")
                if p_:
                    st.markdown(AG.plan_markdown(p_))
                    for wk, md in p_["worksheets"].items():
                        with st.expander(f"📄 {wk} worksheet"): st.markdown(md)
                    if st.button("📥 Send plan for approval"):
                        n = AG.add_to_queue(d, [{"agent": "Remedial planner", "kind": "remedial_plan", "level": "Medium", "student_id": p_["student_id"], "section": p_["section"], "owner": p_["owner"],
                                                 "title": f"{p_['student_name']} ({p_['student_id']}) — {p_['subject']}: {p_['diagnosis']}", "body_en": AG.plan_markdown(p_), "payload": p_, "key": f"plan:{p_['plan_id']}"}])
                        st.success("Plan added to the Approval inbox." if n else "This plan is already waiting or was decided recently.")
            else:
                n_ = st.slider("How many students?", 3, 20, 10)
                if st.button("🤖 Plan for the highest-risk students", type="primary"):
                    r_ = AG.remedial_autoplan(d, n_, owner_filter=owner_f)
                    st.success(f"{len(r_['plans'])} plans drafted · {r_['items_created']} new ones waiting for approval.")
            st.subheader("Plans waiting for approval"); inbox(["remedial_plan"], "rp")
            st.subheader("Active plans — re-check after the next exam")
            rc = AG.recheck_plans(d)
            if rc and ROLE == "teacher": rc = [x for x in rc if x["owner"] == SCOPE]
            if rc:
                df_ = pd.DataFrame([{"Student": f"{x['student_id']} — {x['student_name']}", "Subject": x["subject"], "Diagnosis": x["diagnosis"], "Before": x["baseline"], "Target": x["target"],
                                     "Now": x["now"], "Result": x["outcome"], "Next step": x["next_step"]} for x in rc])
                st.dataframe(df_, hide_index=True, use_container_width=True)
            else:
                st.caption("No active plans have reached their re-check exam yet.")

    # ---- 5. data checks
    if "🧪 Data checks" in T:
        with T["🧪 Data checks"]:
            st.caption("Finds impossible marks, mistyped IDs and suspicious jumps, then asks the teacher who entered them — with a suggested fix. **Approving a fix changes the marks.**")
            if st.button("▶ Run data-health agent", type="primary"):
                with st.spinner("Checking marks…"):
                    r_ = AG.data_health_agent(d, owner_filter=owner_f)
                st.session_state["dh"] = r_["findings"]; st.success(f"{r_['items_created']} new question(s) added.")
            if st.session_state.get("dh") is not None:
                st.dataframe(st.session_state["dh"].rename(columns={"check": "Check", "count": "Found", "status": ""}), hide_index=True, use_container_width=True)
            inbox(["data_check"], "dh")

    # ---- approval inbox
    if "📥 Approval inbox" in T:
        with T["📥 Approval inbox"]:
            inbox(None, "all")
            with st.expander("Recently decided"):
                dq = AG.visible_queue(d, ROLE, SCOPE, status=None)
                dq = dq[dq.status != "Pending"].sort_values("decided_at", ascending=False).head(15)
                st.dataframe(dq[["item_id", "title", "status", "decided_by", "decided_at"]], hide_index=True, use_container_width=True) if len(dq) else st.caption("Nothing decided yet.")

    # ---- audit log
    if "🧾 Agent log" in T:
        with T["🧾 Agent log"]:
            st.caption("Every agent run and every human decision is recorded here.")
            lg = pd.read_csv(d.path("agent_log.csv")).sort_values("time", ascending=False) if os.path.exists(d.path("agent_log.csv")) else pd.DataFrame()
            st.dataframe(lg.head(100), hide_index=True, use_container_width=True) if len(lg) else st.caption("No runs yet.")

# ================================================================== PILOT TOOLKIT
elif page == "🧪 Pilot Toolkit":
    st.title("Pilot Toolkit — testing with a real school")
    t1, t2, t3, t4 = st.tabs(["📋 Checklist", "🕶️ Make data anonymous", "📝 Feedback from users", "📏 Is the prediction accurate?"])
    with t1:
        st.markdown("""
**Before the pilot**
- [ ] Written permission from the Principal (what data, how long, who sees it)
- [ ] One class, last 2–3 exams, marks per subject (per topic if possible)
- [ ] Make the data **anonymous** (tab 2) — keep the ID key file with the school, not with you
- [ ] Load data in **Marks Entry → Import other format**

**During the pilot (30 minutes each: Principal + 2–3 teachers)**
- [ ] Ask them to find: the weakest subject, one student at risk, and what to re-teach — *without help*
- [ ] Write down every place they hesitate or ask "what does this mean?"
- [ ] Ask: **"Did it show you anything you didn't already know?"**

**After the pilot**
- [ ] Record answers in tab 3
- [ ] Check prediction accuracy on real data (tab 4) and report it honestly
- [ ] Fix the top 3 confusions before showing anyone else
""")
    with t2:
        st.caption("Replaces real IDs with codes like S1001 and removes name columns. Download the **key file** separately and give it to the school to keep.")
        f = st.file_uploader("Upload the school's file (CSV/Excel)", type=["csv", "xlsx"], key="anon")
        if f:
            raw = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            idc = st.selectbox("ID / roll-number column", raw.columns, key="an_id")
            guess = [c for c in raw.columns if any(w in c.lower() for w in ["name", "father", "mother", "phone", "mobile", "address", "aadhar", "aadhaar", "email"])]
            ncols = st.multiselect("Columns to remove (names, phone, address…)", [c for c in raw.columns if c != idc], default=[c for c in guess if c != idc])
            an, key_ = B.anonymize(raw, idc, ncols)
            st.dataframe(an.head(8), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Anonymous data (use this)", an.to_csv(index=False), "anonymous_data.csv", type="primary")
            c2.download_button("🔑 ID key (school keeps this)", key_.to_csv(index=False), "ID_KEY_keep_private.csv")
    with t3:
        fb_path = P("pilot_feedback.csv")
        with st.form("pf", clear_on_submit=True):
            c1, c2 = st.columns(2)
            who = c1.selectbox("Who", ["Principal", "HOD", "Teacher", "Management", "Parent"]); nm = c2.text_input("Name (optional)")
            new_info = st.radio("Did it show something they didn't already know?", ["Yes", "Partly", "No"], horizontal=True)
            ease = st.slider("How easy was it to use? (1 = very hard, 5 = very easy)", 1, 5, 3)
            conf = st.text_area("Where did they get confused?"); want = st.text_area("What did they ask for?")
            if st.form_submit_button("Save feedback", type="primary"):
                row = pd.DataFrame([{"date": dt.date.today().isoformat(), "who": who, "name": nm, "new_insight": new_info, "ease": ease, "confusion": conf, "requests": want}])
                (pd.concat([pd.read_csv(fb_path), row]) if os.path.exists(fb_path) else row).to_csv(fb_path, index=False); st.success("Saved.")
        if os.path.exists(fb_path):
            fb = pd.read_csv(fb_path)
            c = st.columns(3)
            c[0].metric("Responses", len(fb)); c[1].metric("Found something new", f"{(fb.new_insight == 'Yes').mean() * 100:.0f}%"); c[2].metric("Ease of use", f"{fb.ease.mean():.1f} / 5")
            st.dataframe(fb, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Download feedback", fb.to_csv(index=False), "pilot_feedback.csv")
    with t4:
        st.caption("A fair test: the system learns from earlier exams and predicts a later one it has **never seen** — exactly how it would be used in real life.")
        if st.button("▶️ Run the accuracy check", type="primary"):
            with st.spinner("Testing…"):
                st.session_state["ev"] = B.evaluate_model_honestly(ss, students, attendance)
        ev = st.session_state.get("ev")
        if ev:
            c = st.columns(3)
            c[0].metric("Students who really failed", ev["actual"])
            c[1].metric("Caught by the system", f"{ev['tp']} of {ev['actual']}", f"{ev['tp'] / max(ev['actual'], 1) * 100:.0f}%")
            c[2].metric("False alarms", f"{ev['fp']} of {ev['flagged']} flagged", help="Flagged but passed")
            card(f"<b>In plain words:</b> out of {ev['actual']} students who failed, the system warned about <b>{ev['tp']}</b> in advance and missed <b>{ev['fn']}</b>. "
                 f"It also flagged <b>{ev['fp']}</b> students who then passed (they still benefit from extra help). "
                 f"A simple rule — 'anyone who failed last exam' — would have caught only <b>{ev['baseline_caught']}</b>.",
                 "grn" if ev["tp"] > ev["baseline_caught"] else "amb")
            show(px.bar(ev["calibration"], x="bin", y="actually_failed", text=ev["calibration"].students.map(lambda n: f"{n} students"),
                        title="When the system says X% risk, how many really failed?", labels={"bin": "Predicted risk", "actually_failed": "% who actually failed"}), 320)
            explain("If the bars rise from left to right, the risk numbers can be trusted. On sample data the result looks good — repeat this on the pilot school's real data and report the real number, even if it is lower.")
            with st.expander("Technical details"):
                st.write(f"AUC {ev['auc']:.3f} · TP {ev['tp']} · FP {ev['fp']} · FN {ev['fn']} · TN {ev['tn']}")

# ================================================================== DATA HEALTH
elif page == "🩺 Data Health":
    st.title("Data Health Check")
    st.caption("Run this after every upload. Wrong data gives wrong conclusions.")
    for h in A.data_health(marks, students):
        with st.expander(f"{h['status']}  {h['check']} — {h['count']}"):
            if h["count"]:
                st.dataframe(h["rows"], hide_index=True, use_container_width=True)
            else:
                st.write("No problems found.")
    st.subheader("Absent students are no longer counted as 0")
    ab = marks.groupby(["student_id", "exam_order", "subject"]).agg(m=("marks", "sum"), mx=("max_marks", "sum")).reset_index()
    old = (ab.m / ab.mx * 100).groupby(ab.exam_order).mean()
    new = ss.groupby("exam_order").score.mean()
    cmp = pd.DataFrame({"exam": EXAMS, "Old way (absent = 0)": old.values, "Correct way (absent left out)": new.reindex(old.index).values})
    show(px.bar(cmp.melt(id_vars="exam"), x="exam", y="value", color="variable", barmode="group", labels={"value": "School average", "exam": ""},
                color_discrete_map={"Old way (absent = 0)": "#8f9bb3", "Correct way (absent left out)": "#4f8ff7"}, title="Effect of counting absences correctly"), 300)
    explain("If a student missed an exam, giving them 0 makes the class look worse than it is and can wrongly flag a teacher. "
            "Absent papers are now left out of averages and failure counts, and listed separately so the school can follow up.")
    st.subheader("Was any exam paper harder or easier than usual?")
    dif = A.exam_difficulty(ss)
    show(px.bar(dif.melt(id_vars=["exam"], value_vars=["raw", "adjusted"], var_name="type"), x="exam", y="value", color="type", barmode="group",
                labels={"value": "School average", "exam": ""}, color_discrete_map={"raw": "#8f9bb3", "adjusted": "#2fbf71"},
                title="School average: actual marks vs adjusted for paper difficulty"), 300)
    for r_ in dif[dif.verdict != "Normal"].itertuples():
        card(f"<b>{r_.exam}</b>: {r_.verdict.lower()} — the whole school scored {abs(r_.shift):.1f} marks {'lower' if r_.shift < 0 else 'higher'} than in a normal exam. "
             "Don't read this as students getting worse. Use the <b>⚖️ Adjust for exam difficulty</b> switch in the sidebar when comparing exams.", "amb")
    explain("When every subject and every section drops in the same exam, the paper was probably harder. The adjusted marks remove that school-wide shift "
            "but keep all differences between students, sections and subjects.")

# ================================================================== SETTINGS
elif page == "⚙️ Settings":
    st.title("Settings")
    if ROLE == "principal":
        st.subheader("School targets")
        with st.form("tg"):
            c = st.columns(4)
            pr = c[0].number_input("Pass rate %", 0, 100, int(targets["pass_rate"]))
            av = c[1].number_input("School average", 0, 100, int(targets["school_avg"]))
            di = c[2].number_input("Distinction %", 0, 100, int(targets["distinction"]))
            at = c[3].number_input("Attendance %", 0, 100, int(targets["attendance"]))
            st.markdown("Subject average targets")
            c = st.columns(len(SUBJECTS))
            sa = {s_: c[i].number_input(s_, 0, 100, int(targets["subject_avg"].get(s_, 60))) for i, s_ in enumerate(SUBJECTS)}
            if st.form_submit_button("Save targets", type="primary"):
                json.dump({"pass_rate": pr, "school_avg": av, "distinction": di, "attendance": at, "subject_avg": sa}, open(P("targets.json"), "w"), indent=2)
                st.success("Targets saved."); st.rerun()
    st.subheader("Change my password")
    with st.form("pw", clear_on_submit=True):
        old = st.text_input("Current password", type="password"); new = st.text_input("New password", type="password"); new2 = st.text_input("Repeat new password", type="password")
        if st.form_submit_button("Change password"):
            us = users(); i = us.index[us.username == U["username"]][0]
            if us.at[i, "password_hash"] != hash_pw(old): st.error("Current password is wrong.")
            elif len(new) < 6 or new != new2: st.error("New passwords must match and be at least 6 characters.")
            else:
                us.at[i, "password_hash"] = hash_pw(new); us.to_csv(P("users.csv"), index=False); st.success("Password changed.")
