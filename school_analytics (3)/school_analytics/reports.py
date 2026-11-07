"""PDF reports: Principal (2 pages), HOD, Teacher one-pager, Parent report card (English + Telugu)."""
import io, os, datetime as dt
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
import analytics as A

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
NAVY, GREY, RED, AMB, GRN, BLUE = (24, 32, 52), (110, 118, 135), (214, 69, 69), (230, 150, 20), (35, 160, 95), (60, 120, 220)
SCOL = {"Good": GRN, "Watch": AMB, "Action needed": RED}

TE = {  # Telugu labels for parent report cards
    "title": "విద్యార్థి ప్రగతి నివేదిక", "name": "విద్యార్థి పేరు", "section": "విభాగం", "exam": "పరీక్ష",
    "subject": "విషయం", "marks": "మార్కులు", "class_avg": "తరగతి సగటు", "status": "స్థితి",
    "average": "సగటు", "percentile": "పర్సంటైల్", "attendance": "హాజరు", "assign": "అసైన్‌మెంట్లు",
    "strengths": "బలాలు", "improve": "మెరుగుపరచుకోవాల్సినవి", "advice": "తల్లిదండ్రులకు సూచనలు",
    "progress": "పరీక్షల వారీగా ప్రగతి", "sign": "తరగతి ఉపాధ్యాయుడు / ప్రిన్సిపాల్ సంతకం",
    "Good": "బాగుంది", "Watch": "గమనించాలి", "Action needed": "సహాయం అవసరం",
    "Mathematics": "గణితం", "Physics": "భౌతిక శాస్త్రం", "Chemistry": "రసాయన శాస్త్రం", "English": "ఆంగ్లం",
    "Computer Science": "కంప్యూటర్ సైన్స్", "Biology": "జీవ శాస్త్రం",
    "adv_att": "హాజరు తక్కువగా ఉంది. దయచేసి ప్రతిరోజూ పాఠశాలకు పంపండి.",
    "adv_assign": "అసైన్‌మెంట్లు సమయానికి పూర్తి చేసేలా చూడండి.",
    "adv_study": "ఇంట్లో రోజుకు కనీసం 2 గంటలు చదువుకునేలా చూడండి.",
    "adv_screen": "మొబైల్ / టీవీ సమయాన్ని రోజుకు 2 గంటలకు తగ్గించండి.",
    "adv_sleep": "రాత్రి కనీసం 7 గంటలు నిద్రపోయేలా చూడండి.",
    "adv_weak": "బలహీనమైన విషయాల కోసం పాఠశాల అదనపు తరగతులకు పంపండి.",
    "adv_good": "మంచి ప్రగతి. ఇలాగే కొనసాగించమని ప్రోత్సహించండి.",
}
EN_ADV = {"adv_att": "Attendance is low. Please send your child to school every day.",
          "adv_assign": "Please make sure assignments are completed on time.",
          "adv_study": "Make sure your child studies at least 2 hours a day at home.",
          "adv_screen": "Reduce mobile / TV time to 2 hours a day.",
          "adv_sleep": "Make sure your child sleeps at least 7 hours.",
          "adv_weak": "Please send your child to the school's extra classes for weak subjects.",
          "adv_good": "Good progress. Encourage your child to keep it up."}


class PDF(FPDF):
    def __init__(self, title, subtitle=""):
        super().__init__("P", "mm", "A4")
        self.add_font("D", "", os.path.join(FONT_DIR, "DejaVuSans.ttf"))
        self.add_font("D", "B", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf"))
        self.add_font("T", "", os.path.join(FONT_DIR, "NotoSansTelugu-Regular.ttf"))
        self.add_font("T", "B", os.path.join(FONT_DIR, "NotoSansTelugu-Bold.ttf"))
        self.set_fallback_fonts(["T"], exact_match=False)
        self.set_text_shaping(True)
        self.t, self.st = title, subtitle
        self.set_auto_page_break(True, 15)
        self.set_margins(14, 14, 14)

    def header(self):
        self.set_fill_color(*NAVY); self.rect(0, 0, 210, 24, "F")
        self.set_fill_color(242, 200, 17); self.rect(0, 24, 210, 1.2, "F")
        self.set_xy(14, 6); self.set_text_color(255, 255, 255); self.set_font("D", "B", 15)
        self.cell(0, 7, self.t)
        self.set_xy(14, 14); self.set_font("D", "", 9); self.set_text_color(200, 208, 222)
        self.cell(0, 5, self.st)
        self.set_xy(14, 30); self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-11); self.set_font("D", "", 7.5); self.set_text_color(*GREY)
        self.cell(0, 5, f"School Performance Analytics · generated {dt.date.today():%d %b %Y} · page {self.page_no()}", align="C")

    def h2(self, txt):
        self.ln(2); self.set_font("D", "B", 11.5); self.set_text_color(*NAVY); self.cell(0, 7, txt, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(220, 224, 232); self.line(14, self.get_y(), 196, self.get_y()); self.ln(1.5)
        self.set_text_color(0, 0, 0)

    def para(self, txt, size=9.5, color=(30, 30, 30), bold=False):
        self.set_font("D", "B" if bold else "", size); self.set_text_color(*color)
        self.multi_cell(0, 5, txt, new_x="LMARGIN", new_y="NEXT"); self.set_text_color(0, 0, 0)

    def dot(self, color, x=None, y=None, r=1.8):
        x = self.get_x() + r if x is None else x; y = self.get_y() + 3 if y is None else y
        self.set_fill_color(*color); self.ellipse(x - r, y - r, 2 * r, 2 * r, "F")

    def table(self, header, rows, widths, colors=None, size=8.5, row_h=6.2):
        self.set_font("D", "B", size); self.set_fill_color(236, 239, 245); self.set_text_color(*NAVY)
        for h, w in zip(header, widths):
            self.cell(w, row_h, h, border=0, fill=True)
        self.ln(row_h); self.set_font("D", "", size); self.set_text_color(20, 20, 20)
        for i, r in enumerate(rows):
            if self.get_y() > 275: self.add_page()
            fill = i % 2 == 1; self.set_fill_color(248, 249, 252)
            for j, (v, w) in enumerate(zip(r, widths)):
                if colors and colors[i] and j == colors[i][0]:
                    y = self.get_y(); x = self.get_x()
                    self.cell(w, row_h, "", fill=fill); self.set_x(x)
                    self.dot(colors[i][1], x + 2.5, y + row_h / 2)
                    self.set_x(x + 6); self.cell(w - 6, row_h, str(v))
                else:
                    self.cell(w, row_h, str(v), fill=fill)
            self.ln(row_h)

    def bar(self, label, pct, now_txt, color):
        y = self.get_y(); self.set_font("D", "", 9); self.cell(55, 6, label)
        x = self.get_x(); self.set_fill_color(230, 233, 240); self.rect(x, y + 1.5, 90, 3.5, "F")
        self.set_fill_color(*color); self.rect(x, y + 1.5, 90 * min(pct, 100) / 100, 3.5, "F")
        self.set_x(x + 93); self.cell(0, 6, now_txt, new_x="LMARGIN", new_y="NEXT")

    def problem_box(self, i, p):
        if self.get_y() > 245: self.add_page()
        y0 = self.get_y(); self.set_fill_color(252, 243, 243) if i == 0 else self.set_fill_color(253, 248, 236)
        lines = [("Problem", p["problem"]), ("Why", p["reason"]), ("Action", p["action"]), ("Owner / deadline", f"{p['owner']}  ·  by {p['deadline']}")]
        self.set_x(18)
        for k, v in lines:
            self.set_x(18); self.set_font("D", "B", 8.5); self.set_text_color(*NAVY); self.cell(30, 5, k)
            self.set_font("D", "", 8.5); self.set_text_color(25, 25, 25); self.multi_cell(148, 5, v, new_x="LMARGIN", new_y="NEXT")
        y1 = self.get_y(); self.set_fill_color(*(RED if i == 0 else AMB)); self.rect(14, y0, 1.6, y1 - y0, "F"); self.ln(2.5)

    def image_fig(self, fig, w=182):
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=170, bbox_inches="tight"); plt.close(fig); b.seek(0)
        self.image(b, w=w); self.ln(2)


def _style():
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa3b5",
                         "axes.titleweight": "bold", "axes.titlesize": 10, "font.family": "DejaVu Sans"})


def _out(pdf):
    return bytes(pdf.output())


# ---------------------------------------------------------------- principal
def principal_pdf(ss, students, teachers, marks, exam, targets, last_year, actions, risk_prob):
    _style()
    status = A.subject_status(ss, students, exam, targets, last_year)
    probs = A.principal_problems(ss, students, teachers, marks, exam, risk_prob)
    exams = ss.sort_values("exam_order").exam.unique().tolist(); ei = exams.index(exam)
    k = A.exam_kpis(ss, students, exam); kp = A.exam_kpis(ss, students, exams[ei - 1]) if ei else None
    pdf = PDF("Principal's Exam Summary", f"{exam} · {len(students)} students · {students.section.nunique()} sections")
    pdf.add_page()
    pdf.set_fill_color(240, 244, 252); pdf.set_font("D", "B", 10.5); pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 6.5, A.headline(status, probs, k, kp), fill=True, padding=3, new_x="LMARGIN", new_y="NEXT"); pdf.ln(2)

    pdf.h2("Are we reaching our goals?")
    for r in A.target_progress(ss, students, exam, targets).itertuples():
        col = GRN if r.progress >= 100 else AMB if r.progress >= 90 else RED
        pdf.bar(r.Goal, r.progress, f"{r.Now:.1f}{r.unit} of {r.Target}{r.unit} target ({r.progress:.0f}%)", col)

    pdf.h2("Subject health")
    rows, cols = [], []
    for r in status.itertuples():
        rows.append([r.subject, f"{r.avg:.1f}", f"{r.fail:.1f}%", A.change_words(r.vs_last_exam), A.change_words(r.vs_last_year), r.status])
        cols.append((5, SCOL[r.status]))
    pdf.table(["Subject", "Average", "Failing", "vs last exam", "vs last year", "Status"], rows, [34, 18, 18, 40, 40, 32], cols)
    pdf.ln(1); pdf.para("Status: Green = on target · Amber = slightly below target, watch · Red = needs action now.", 7.5, GREY)

    pdf.h2("Top problems and what to do")
    for i, p in enumerate(probs[:3]):
        pdf.problem_box(i, p)

    # page 2: charts + actions
    pdf.add_page(); pdf.h2("Subject averages: this exam vs last exam vs last year")
    fig, ax = plt.subplots(figsize=(9, 3.1))
    x = np.arange(len(status)); w = .27
    prev = (status.avg - status.vs_last_exam).values
    ly = (status.avg - status.vs_last_year).values
    ax.bar(x - w, ly, w, label="Last year", color="#c3c9d6"); ax.bar(x, prev, w, label="Last exam", color="#8fb3f5")
    ax.bar(x + w, status.avg, w, label=exam, color=[{"Good": "#23a05f", "Watch": "#e69614", "Action needed": "#d64545"}[s] for s in status.status])
    ax.axhline(targets["school_avg"], ls="--", color="#555", lw=1); ax.text(len(x) - .5, targets["school_avg"] + .8, "target", fontsize=7, ha="right")
    ax.set_xticks(x, status.subject); ax.set_ylim(30, 80); ax.legend(frameon=False, ncol=3, loc="upper left")
    pdf.image_fig(fig)
    pdf.h2("Where is the problem? Section × subject average")
    e = ss[ss.exam == exam].merge(students[["student_id", "section"]], on="student_id")
    hm = e.pivot_table(index="section", columns="subject", values="score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 2.6)); im = ax.imshow(hm.values, cmap="RdYlGn", vmin=45, vmax=75, aspect="auto")
    ax.set_xticks(range(hm.shape[1]), hm.columns); ax.set_yticks(range(hm.shape[0]), ["Section " + s for s in hm.index])
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]): ax.text(j, i, f"{hm.values[i, j]:.0f}", ha="center", va="center", fontsize=8.5, weight="bold")
    ax.spines[:].set_visible(False); pdf.image_fig(fig)
    pdf.para("Read across a row: if only one box is red, that subject's teaching in that section is the issue. If a whole column is red, the subject is hard for everyone.", 8, GREY)
    pdf.h2("Action tracker — did our actions work?")
    if actions is None or actions.empty:
        pdf.para("No actions recorded yet.")
    else:
        rows = []
        for a in actions.to_dict("records"):
            im_ = A.action_impact(ss, a)
            res = "—" if im_ is None else (f"Worked: +{im_['net']:.1f} marks vs others" if im_["worked"] else f"No clear effect yet ({im_['net']:+.1f})")
            rows.append([a["action"][:48], a["owner"], a["deadline"], a["status"], res])
        pdf.table(["Action", "Owner", "Deadline", "Status", "Result"], rows, [66, 32, 22, 22, 40], size=7.8)
    return _out(pdf)


# ---------------------------------------------------------------- HOD
def hod_pdf(ss, students, teachers, marks, exam, subject, targets):
    _style()
    s = ss[(ss.subject == subject) & (ss.exam == exam)].merge(students[["student_id", "section"]], on="student_id")
    pdf = PDF(f"HOD Report — {subject}", f"{exam} · {s.student_id.nunique()} students")
    pdf.add_page()
    avg, fail = s.score.mean(), (s.score < A.PASS).mean() * 100
    st = A.status_of(avg, fail, targets["subject_avg"].get(subject, 60))
    pdf.para(f"Subject average {avg:.1f} (target {targets['subject_avg'].get(subject, 60)}) · {fail:.1f}% failing · Status: {st}", 10.5, NAVY, True)
    pdf.h2("Topics — which ones need re-teaching?")
    tm = marks[(marks.subject == subject) & (marks.exam == exam)].merge(students[["student_id", "section"]], on="student_id")
    hm = tm.pivot_table(index="topic", columns="section", values="marks", aggfunc="mean") / 25 * 100
    fig, ax = plt.subplots(figsize=(9, 2.4)); ax.imshow(hm.values, cmap="RdYlGn", vmin=35, vmax=80, aspect="auto")
    ax.set_yticks(range(len(hm)), hm.index); ax.set_xticks(range(hm.shape[1]), ["Section " + c for c in hm.columns])
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]): ax.text(j, i, f"{hm.values[i, j]:.0f}%", ha="center", va="center", weight="bold")
    ax.spines[:].set_visible(False); pdf.image_fig(fig)
    weak = hm.mean(1).sort_values()
    pdf.para(f"Hardest topic: {weak.index[0]} ({weak.iloc[0]:.0f}% scored on average). If it is low in every section, revise the lesson plan; if low in one section only, support that teacher.", 9)
    pdf.h2("Teachers — compared fairly (value-added)")
    va = A.value_added(ss, students, teachers, exam); va = va[va.subject == subject]
    rows, cols = [], []
    for r in va.itertuples():
        secs = ", ".join(sorted(s[s.teacher_id == r.teacher_id].section.unique()))
        verdict = "Above expected" if r.value_added > 2 else "Below expected" if r.value_added < -2 else "As expected"
        rows.append([r.teacher_name, secs, r.experience_yrs, f"{r.avg_score:.1f}", f"{r.value_added:+.1f}", verdict])
        cols.append((5, GRN if verdict == "Above expected" else RED if verdict == "Below expected" else GREY))
    pdf.table(["Teacher", "Sections", "Exp (yrs)", "Average", "Marks vs expected", "Verdict"], rows, [50, 22, 18, 20, 32, 40], cols)
    pdf.para("'Marks vs expected' compares each class with what its students were expected to score from last year's results — fairer than raw averages.", 7.5, GREY)
    pdf.h2("Question paper check")
    w = tm.pivot_table(index="student_id", columns="topic", values="marks")
    tot = w.sum(1); hi, lo = tot >= tot.quantile(.73), tot <= tot.quantile(.27)
    disc = ((w[hi].mean() - w[lo].mean()) / 25).round(2)
    rows = [[t, f"{w[t].mean() / 25 * 100:.0f}%", f"{disc[t]:.2f}", "Good" if disc[t] >= .3 else "OK" if disc[t] >= .2 else "Revise this section"] for t in w.columns]
    pdf.table(["Section of paper", "Avg scored", "Separates strong/weak", "Verdict"], rows, [50, 30, 50, 52])
    return _out(pdf)


# ---------------------------------------------------------------- teacher
def teacher_pdf(ss, students, teachers, marks, exam, teacher_id, risk_prob, reasons):
    _style()
    t = teachers.set_index("teacher_id").loc[teacher_id]
    mine = ss[(ss.teacher_id == teacher_id) & (ss.exam == exam)].merge(students[["student_id", "student_name", "section", "attendance_pct"]], on="student_id")
    school = ss[(ss.subject == t.subject) & (ss.exam == exam)]
    pdf = PDF(f"Teacher Sheet — {t.teacher_name}", f"{t.subject} · Sections {', '.join(sorted(mine.section.unique()))} · {exam}")
    pdf.add_page()
    pdf.para(f"My class average {mine.score.mean():.1f}  (school {school.score.mean():.1f})   ·   failing {int((mine.score < A.PASS).sum())}   ·   distinctions {int((mine.score >= 75).sum())}", 10.5, NAVY, True)
    pdf.h2("Topics: my classes vs rest of school")
    tm = marks[(marks.subject == t.subject) & (marks.exam == exam)]
    a = tm[tm.teacher_id == teacher_id].groupby("topic").marks.mean() / 25 * 100
    b = tm[tm.teacher_id != teacher_id].groupby("topic").marks.mean() / 25 * 100
    fig, ax = plt.subplots(figsize=(9, 2.6)); x = np.arange(len(a))
    ax.bar(x - .2, a.values, .4, label="My classes", color="#f2c811"); ax.bar(x + .2, b.reindex(a.index).values, .4, label="Rest of school", color="#8fb3f5")
    ax.set_xticks(x, a.index); ax.set_ylabel("% scored"); ax.legend(frameon=False); pdf.image_fig(fig)
    w = a.sort_values()
    pdf.para(f"Re-teach first: {w.index[0]} ({w.iloc[0]:.0f}%) and {w.index[1]} ({w.iloc[1]:.0f}%).", 10, RED, True)
    pdf.h2("Students who need my attention")
    mine["risk"] = mine.student_id.map(risk_prob) * 100
    lst = mine[(mine.score < 45) | (mine.risk >= 30)].sort_values("risk", ascending=False).head(28)
    rows = [[r.student_id, r.student_name, r.section, f"{r.score:.0f}", f"{r.attendance_pct:.0f}%", f"{r.risk:.0f}%", (reasons.get(r.student_id, "") or "")[:46]] for r in lst.itertuples()]
    pdf.table(["ID", "Name", "Sec", "Score", "Attend.", "Fail risk", "Why"], rows, [16, 34, 10, 14, 16, 16, 76], size=7.6, row_h=5.6)
    return _out(pdf)


# ---------------------------------------------------------------- parent
def advice_keys(s, scores):
    k = []
    if s.attendance_pct < 80: k.append("adv_att")
    if s.assignment_submission_pct < 70: k.append("adv_assign")
    if s.study_hours_per_day < 1.5: k.append("adv_study")
    if s.screen_time_hours > 4.5: k.append("adv_screen")
    if s.sleep_hours < 6: k.append("adv_sleep")
    if (scores < 45).any(): k.append("adv_weak")
    return k or ["adv_good"]


def parent_pdf(ss, students, marks, exam, sid, risk=None):
    _style()
    s = students.set_index("student_id").loc[sid]
    me = ss[ss.student_id == sid]
    now = me[me.exam == exam].set_index("subject").score
    cls = ss[(ss.exam == exam) & ss.student_id.isin(students[students.section == s.section].student_id)].groupby("subject").score.mean()
    allavg = ss[ss.exam == exam].groupby("student_id").score.mean()
    pct = (allavg < allavg[sid]).mean() * 100
    pdf = PDF("Student Progress Report", f"{TE['title']}  ·  {exam}")
    pdf.add_page()
    pdf.set_font("D", "B", 11); pdf.set_text_color(*NAVY)
    pdf.cell(0, 6, f"{s.student_name}  ({sid})  ·  Section / {TE['section']} {s.section}", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
    # KPI boxes
    boxes = [(f"Average / {TE['average']}", f"{now.mean():.1f}"), (f"Percentile / {TE['percentile']}", f"{pct:.0f}"),
             (f"Attendance / {TE['attendance']}", f"{s.attendance_pct:.0f}%"), (f"Assignments / {TE['assign']}", f"{s.assignment_submission_pct:.0f}%")]
    y = pdf.get_y()
    for i, (lab, val) in enumerate(boxes):
        x = 14 + i * 46; pdf.set_fill_color(240, 244, 252); pdf.rect(x, y, 43, 17, "F")
        pdf.set_xy(x + 2, y + 1.5); pdf.set_font("D", "", 7.5); pdf.set_text_color(*GREY); pdf.cell(40, 5, lab)
        pdf.set_xy(x + 2, y + 7); pdf.set_font("D", "B", 14); pdf.set_text_color(*NAVY); pdf.cell(40, 8, val)
    pdf.set_xy(14, y + 21)
    pdf.h2(f"Marks / {TE['marks']}")
    rows, cols = [], []
    for subj, sc in now.items():
        stt = "Good" if sc >= 60 else "Watch" if sc >= 45 else "Action needed"
        lab = {"Good": "Good", "Watch": "Watch", "Action needed": "Needs help"}[stt]
        rows.append([f"{subj} / {TE.get(subj, subj)}", f"{sc:.0f} / 100", f"{cls[subj]:.0f}", f"{lab} / {TE[stt]}"])
        cols.append((3, SCOL[stt]))
    pdf.table([f"Subject / {TE['subject']}", f"Marks / {TE['marks']}", f"Class avg / {TE['class_avg']}", f"Status / {TE['status']}"], rows, [58, 28, 38, 58], cols, size=8.3, row_h=6.4)
    pdf.h2(f"Progress across exams / {TE['progress']}")
    fig, ax = plt.subplots(figsize=(9, 2.1))
    piv = me.pivot_table(index="exam_order", columns="subject", values="score")
    for c in piv.columns: ax.plot(piv.index, piv[c], marker="o", label=c)
    ax.set_xticks(piv.index, me.sort_values("exam_order").exam.unique()); ax.set_ylim(0, 100); ax.axhline(A.PASS, ls="--", color="#d64545", lw=.8)
    ax.legend(frameon=False, ncol=6, fontsize=7, loc="lower center", bbox_to_anchor=(.5, -.35)); pdf.image_fig(fig, 165)
    tp = marks[(marks.student_id == sid) & (marks.exam == exam)].assign(p=lambda x: x.marks / x.max_marks * 100).sort_values("p")
    pdf.h2(f"Strengths / {TE['strengths']}")
    pdf.para(", ".join(f"{r.topic} ({TE.get(r.subject, r.subject)}) {r.p:.0f}%" for r in tp.tail(3).iloc[::-1].itertuples()), 9.5, (20, 110, 60))
    pdf.h2(f"Needs improvement / {TE['improve']}")
    pdf.para(", ".join(f"{r.topic} ({TE.get(r.subject, r.subject)}) {r.p:.0f}%" for r in tp.head(3).itertuples()), 9.5, RED)
    pdf.h2(f"Advice for parents / {TE['advice']}")
    for kk in advice_keys(s, now):
        pdf.para(f"•  {EN_ADV[kk]}", 9.2); pdf.para(f"    {TE[kk]}", 9.2, (60, 60, 60))
    pdf.ln(5); pdf.set_font("D", "", 8.5); pdf.set_text_color(*GREY)
    pdf.cell(0, 5, f"_______________________      Signature of class teacher / Principal  ·  {TE['sign']}")
    return _out(pdf)


def parent_pdfs_zip(ss, students, marks, exam, ids):
    import zipfile
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as z:
        for sid in ids:
            z.writestr(f"report_card_{sid}.pdf", parent_pdf(ss, students, marks, exam, sid))
    return b.getvalue()


# ---------------------------------------------------------------- excel template
def marks_template(students_df, subject, topics, exam, teacher_name, max_marks=25):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook(); ws = wb.active; ws.title = "Marks"
    hdr = ["student_id", "student_name", "section"] + [f"{t} (out of {max_marks})" for t in topics]
    ws.append(hdr)
    for r in students_df.sort_values(["section", "student_id"]).itertuples():
        ws.append([r.student_id, r.student_name, r.section] + [None] * len(topics))
    fill = PatternFill("solid", fgColor="1D2639"); inp = PatternFill("solid", fgColor="FFF8DC")
    for c in ws[1]: c.font = Font(bold=True, color="FFFFFF"); c.fill = fill; c.alignment = Alignment(wrap_text=True)
    for col, w in zip("ABCDEFGHIJ", [12, 24, 9] + [16] * len(topics)): ws.column_dimensions[col].width = w
    dv = DataValidation(type="decimal", operator="between", formula1="0", formula2=str(max_marks), allow_blank=True,
                        showErrorMessage=True, errorTitle="Invalid marks", error=f"Enter a number between 0 and {max_marks} (or leave blank if absent)")
    ws.add_data_validation(dv)
    last = ws.max_row
    for j in range(len(topics)):
        col = chr(ord("D") + j); dv.add(f"{col}2:{col}{last}")
        for i in range(2, last + 1): ws[f"{col}{i}"].fill = inp
    ws.freeze_panes = "D2"
    info = wb.create_sheet("Instructions")
    for line in [f"Marks entry — {subject} — {exam}", f"Teacher: {teacher_name}", "",
                 "1. Type marks only in the yellow cells.", f"2. Each topic is out of {max_marks}.",
                 "3. If a student was absent, type AB or leave the cell blank.",
                 "4. Do not change student IDs or column names.", "5. Save and upload this file in the app (Data Entry page)."]:
        info.append([line])
    info["A1"].font = Font(bold=True, size=13); info.column_dimensions["A"].width = 80
    info.sheet_properties.tabColor = "F2C811"
    ws.title = "Marks"; wb.move_sheet("Instructions", offset=-1); wb.active = 1
    meta = wb.create_sheet("meta"); meta.append(["subject", subject]); meta.append(["exam", exam]); meta.sheet_state = "hidden"
    b = io.BytesIO(); wb.save(b); return b.getvalue()
