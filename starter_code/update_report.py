#!/usr/bin/env python3
"""
update_report.py
================
Updates InferTutor_Model_Comparison_Report.pdf with the answer quality
evaluation results from eval_responses.json.

Changes made:
  1. Page 13 (Section 6.4): overlays the "Important caveat" box with an
     update note referencing Section 9.
  2. Page 15 (Section 8 Conclusion): overlays updated conclusion text that
     incorporates the quality evaluation outcome.
  3. Appends new pages for Section 9: Answer Quality Evaluation.

Usage:
  pip install pypdf pdfplumber reportlab
  python update_report.py

Output:
  InferTutor_Model_Comparison_Report_v2.pdf  (same folder as input)
"""

import io
import os
import sys

try:
    import pdfplumber
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, PageBreak, HRFlowable, KeepTogether
    )
    from reportlab.pdfgen import canvas as rl_canvas
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run:  pip install pypdf pdfplumber reportlab")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)  # one level up from starter_code
INPUT_PDF  = os.path.join(REPORT_DIR, "InferTutor_Model_Comparison_Report.pdf")
OUTPUT_PDF = os.path.join(REPORT_DIR, "InferTutor_Model_Comparison_Report_v2.pdf")

# ---------------------------------------------------------------------------
# Colour palette — matches existing PDF styling
# ---------------------------------------------------------------------------
NAVY      = colors.Color(30/255,  45/255,  75/255)   # dark header bg
BLUE      = colors.Color(37/255,  99/255,  235/255)  # section heading
LBLUE_BG  = colors.Color(219/255, 234/255, 254/255)  # light blue box bg
LBLUE_BD  = colors.Color(147/255, 197/255, 253/255)  # light blue border
GREEN_BG  = colors.Color(240/255, 253/255, 244/255)  # green box bg
GREEN_BD  = colors.Color(34/255,  197/255,  94/255)  # green box border
GREEN_TXT = colors.Color(21/255,  128/255,  61/255)  # green text
LGRAY     = colors.Color(249/255, 250/255, 251/255)  # alternating row bg
WHITE     = colors.white
BLACK     = colors.black
GRAY      = colors.Color(107/255, 114/255, 128/255)

PAGE_W, PAGE_H = A4   # 595.28 x 841.89 points
MARGIN_L = 56.7       # 2 cm left margin (in points)
MARGIN_R = PAGE_W - 56.7
CONTENT_W = MARGIN_R - MARGIN_L

# ---------------------------------------------------------------------------
# Hardcoded evaluation results (from eval_responses.json)
# ---------------------------------------------------------------------------
RESULTS = {
    "overall": {
        "qwen_avg": 2.17, "mistral_avg": 1.50,
        "qwen_pct": 72.3, "mistral_pct": 50.0,
    },
    "qwen_dist":    {"3": 40, "2": 42, "1": 13, "0": 5},
    "mistral_dist": {"3":  7, "2": 54, "1": 21, "0": 18},
    "categories": [
        ("KV Cache",                        15, 2.67, 2.27),
        ("TTFT & ITL Diagnosis",            15, 2.07, 1.40),
        ("Batching & Scheduling",           15, 2.20, 1.73),
        ("GPU Memory & Quantisation",       15, 2.27, 1.47),
        ("vLLM Flags & Configuration",      15, 2.20, 1.33),
        ("Replicas vs Tensor Parallelism",  10, 2.10, 1.20),
        ("Chunked Prefill & Prefix Caching",15, 1.67, 1.00),
    ],
}


# ===========================================================================
# 1. PROBE: find element positions on pages 13 and 15
# ===========================================================================
def probe_positions(pdf_path):
    """
    Use pdfplumber to find the top-Y coordinate (from page top, in points)
    of key text anchors on pages 13 and 15.
    Returns dict with keys: caveat_top, conclusion_para_top, resume_top.
    """
    positions = {}
    with pdfplumber.open(pdf_path) as pdf:
        # --- Page 13 (index 12): find "Important" (start of caveat box) ---
        p13 = pdf.pages[12]
        words13 = p13.extract_words()
        for w in words13:
            if w["text"] == "Important":
                positions["caveat_top"] = w["top"]
                break
        if "caveat_top" not in positions:
            positions["caveat_top"] = 710.0   # fallback estimate

        # --- Page 15 (index 14): find "The" in "The optimal production" ---
        p15 = pdf.pages[14]
        words15 = p15.extract_words()
        # Look for "optimal" (part of "The optimal production configuration")
        for w in words15:
            if w["text"] == "optimal":
                positions["conclusion_para_top"] = w["top"]
                break
        if "conclusion_para_top" not in positions:
            positions["conclusion_para_top"] = 570.0

        # Look for "Resume-ready" box
        for w in words15:
            if w["text"] == "Resume-ready":
                positions["resume_top"] = w["top"]
                break
        if "resume_top" not in positions:
            positions["resume_top"] = 680.0

    return positions


# ===========================================================================
# 2. OVERLAY: page 13 — replace caveat box with update note
# ===========================================================================
def make_page13_overlay(caveat_top_plumber):
    """
    Creates a one-page PDF (in memory) that whites-out the old yellow
    caveat box and draws a green update box in its place.

    caveat_top_plumber: pdfplumber Y coordinate (from page top) of the
                        first word in the caveat box.
    """
    # Convert to reportlab coords (from page bottom)
    caveat_top_rl    = PAGE_H - caveat_top_plumber + 4   # slight padding
    caveat_bottom_rl = 42                                 # near page bottom

    box_h = caveat_top_rl - caveat_bottom_rl

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)

    # --- White-out the old yellow caveat box ---
    c.setFillColor(WHITE)
    c.setStrokeColor(WHITE)
    c.rect(MARGIN_L - 2, caveat_bottom_rl - 2,
           CONTENT_W + 4, box_h + 4, fill=1, stroke=0)

    # --- Draw green update box ---
    c.setFillColor(GREEN_BG)
    c.setStrokeColor(GREEN_BD)
    c.setLineWidth(1)
    c.rect(MARGIN_L, caveat_bottom_rl, CONTENT_W, box_h, fill=1, stroke=1)

    # Bold heading
    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(GREEN_TXT)
    heading_y = caveat_top_rl - 16
    c.drawString(MARGIN_L + 8, heading_y,
                 "Update — quality evaluation completed: see Section 9.")

    # Body text
    c.setFont("Helvetica", 8.5)
    c.setFillColor(BLACK)
    lines = [
        "A 100-question domain knowledge evaluation was conducted after the serving experiments,",
        "covering seven inference engineering knowledge categories. Qwen3-VL-4B scored 2.17/3.00",
        "vs Mistral-7B’s 1.50/3.00 — a 45% quality advantage. Both serving performance and answer",
        "quality favour Qwen3-VL-4B. The deployment recommendation in Section 6.1 stands.",
    ]
    text_y = heading_y - 16
    for line in lines:
        c.drawString(MARGIN_L + 8, text_y, line)
        text_y -= 13

    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


# ===========================================================================
# 3. OVERLAY: page 15 — update conclusion with quality eval paragraph
# ===========================================================================
def make_page15_overlay(conclusion_para_top, resume_top):
    """
    Whites out from "The optimal production..." down to the bottom of page,
    then redraws that section with an added quality-eval paragraph and
    updated resume box.
    """
    # Convert pdfplumber tops to reportlab Y (from bottom)
    cover_top_rl    = PAGE_H - conclusion_para_top + 6
    cover_bottom_rl = 40

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)

    # --- White-out ---
    c.setFillColor(WHITE)
    c.setStrokeColor(WHITE)
    c.rect(MARGIN_L - 2, cover_bottom_rl - 2,
           CONTENT_W + 4, (cover_top_rl - cover_bottom_rl) + 4, fill=1, stroke=0)

    # --- "The optimal production..." paragraph ---
    c.setFont("Helvetica", 9.5)
    c.setFillColor(BLACK)
    y = cover_top_rl - 4

    para1 = [
        "The optimal production configuration is Qwen3-VL-4B at 100 concurrent users per H100, with",
        "autoscaling triggered at 80 users to maintain headroom. For larger deployments, horizontal",
        "scaling with additional H100 replicas is recommended over increasing concurrency on a single GPU.",
    ]
    for line in para1:
        c.drawString(MARGIN_L, y, line)
        y -= 14

    y -= 6

    # --- NEW paragraph: quality evaluation outcome ---
    para2 = [
        "A subsequent 100-question domain knowledge evaluation (Section 9) confirmed that Qwen3-VL-4B",
        "also leads on answer quality: 2.17/3.00 vs Mistral’s 1.50/3.00, a 45% advantage. Qwen produced",
        "fully correct answers on 40% of questions vs 7% for Mistral, and hallucinated on only 5% of",
        "questions vs 18% for Mistral. Both serving performance and knowledge quality align on the same",
        "conclusion: Qwen3-VL-4B is the deployment choice.",
    ]
    for line in para2:
        c.drawString(MARGIN_L, y, line)
        y -= 14

    y -= 10

    # --- Resume-ready box (updated) ---
    resume_box_h = 90
    resume_box_y = y - resume_box_h

    c.setFillColor(LBLUE_BG)
    c.setStrokeColor(LBLUE_BD)
    c.setLineWidth(1)
    c.rect(MARGIN_L, resume_box_y, CONTENT_W, resume_box_h, fill=1, stroke=1)

    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(BLACK)
    c.drawString(MARGIN_L + 8, resume_box_y + resume_box_h - 16, "Resume-ready summary:")

    c.setFont("Helvetica", 8.5)
    resume_lines = [
        "Conducted a systematic LLM serving performance evaluation comparing two open-source models",
        "(4B vs 7B parameters) across four concurrency levels on H100 GPU infrastructure. Identified",
        "a memory bandwidth saturation failure mode in the larger model at 150 concurrent users using",
        "ITL p95 monitoring. Complemented with a 100-question domain knowledge evaluation confirming",
        "Qwen3-VL-4B’s superiority on both serving efficiency (38% higher composite score) and answer",
        "quality (2.17 vs 1.50 out of 3, 45% advantage). Recommended deployment configuration based",
        "on composite score, latency stability, GPU efficiency, and domain accuracy.",
    ]
    ry = resume_box_y + resume_box_h - 32
    for line in resume_lines:
        c.drawString(MARGIN_L + 8, ry, line)
        ry -= 11.5

    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


# ===========================================================================
# 4. SECTION 9 pages — full quality evaluation section
# ===========================================================================
def make_section9():
    """
    Returns a BytesIO containing the Section 9 PDF pages built with
    reportlab Platypus.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontName="Helvetica-Bold", fontSize=14,
                        textColor=BLACK, spaceAfter=10, spaceBefore=0)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontName="Helvetica-Bold", fontSize=11,
                        textColor=BLUE, spaceAfter=8, spaceBefore=12)
    BODY = ParagraphStyle("Body", parent=styles["Normal"],
                          fontName="Helvetica", fontSize=9.5,
                          leading=14, spaceAfter=8,
                          alignment=TA_JUSTIFY)
    BULLET = ParagraphStyle("Bullet", parent=BODY,
                            leftIndent=12, spaceAfter=5)

    def tbl(data, col_widths, extra_styles=None):
        base = [
            ("BACKGROUND",  (0, 0), (-1,  0),  NAVY),
            ("TEXTCOLOR",   (0, 0), (-1,  0),  WHITE),
            ("FONTNAME",    (0, 0), (-1,  0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1),  9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LGRAY]),
            ("GRID",        (0, 0), (-1, -1),  0.5,
             colors.Color(0.8, 0.8, 0.8)),
            ("TOPPADDING",  (0, 0), (-1, -1),  5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1),  8),
            ("RIGHTPADDING",(0, 0), (-1, -1),  8),
        ]
        if extra_styles:
            base.extend(extra_styles)
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle(base))
        return t

    R = RESULTS
    story = []

    # -----------------------------------------------------------------------
    # 9. Header
    # -----------------------------------------------------------------------
    story.append(Paragraph("9. Answer Quality Evaluation", H1))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=GRAY, spaceAfter=10))

    # -----------------------------------------------------------------------
    # 9.1 Overview
    # -----------------------------------------------------------------------
    story.append(Paragraph("9.1 Evaluation Overview", H2))
    story.append(Paragraph(
        "Following the serving performance experiments documented in Sections 3–5, a structured "
        "answer quality evaluation was conducted to address the caveat raised in Section 6.4. "
        "A set of 100 questions spanning seven knowledge categories core to LLM inference engineering "
        "was posed to both models. Each response was scored independently by a human evaluator on a "
        "0–3 rubric against a reference answer.", BODY))

    rubric_data = [
        ["Score", "Meaning"],
        ["3", "Fully correct — accurate, complete, no hallucinations"],
        ["2", "Partially correct — mostly right but missing key detail or minor inaccuracy"],
        ["1", "Mostly wrong — fundamental misunderstanding or significant hallucination"],
        ["0", "Completely wrong or refused to answer"],
    ]
    story.append(tbl(rubric_data, [1.2*cm, 13.5*cm],
                     [("ALIGN", (0, 0), (0, -1), "CENTER")]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "The 100 questions were distributed across seven categories: KV Cache (15), "
        "TTFT &amp; ITL Diagnosis (15), Batching &amp; Scheduling (15), GPU Memory &amp; Quantisation (15), "
        "vLLM Flags &amp; Configuration (15), Replicas vs Tensor Parallelism (10), and "
        "Chunked Prefill &amp; Prefix Caching (15). All questions were drawn from the InferTutor "
        "domain curriculum and graded using the reference answers from eval_responses.json.", BODY))

    # -----------------------------------------------------------------------
    # 9.2 Overall Results
    # -----------------------------------------------------------------------
    story.append(Paragraph("9.2 Overall Results", H2))
    story.append(Paragraph(
        f"Qwen3-VL-4B achieved an overall average score of <b>2.17 / 3.00 (72.3%)</b> across all "
        f"100 questions. Mistral-7B-Instruct-v0.3 achieved <b>1.50 / 3.00 (50.0%)</b>. "
        f"Qwen leads by <b>0.67 points — a 45% relative advantage</b> on per-question accuracy.", BODY))

    overall_data = [
        ["Model", "Avg Score", "% of Max",
         "Fully Correct (3)", "Partial (2)", "Mostly Wrong (1)", "Zero (0)"],
        ["Qwen3-VL-4B",
         "2.17 / 3.00", "72.3%",
         "40  (40%)", "42  (42%)", "13  (13%)", "5  (5%)"],
        ["Mistral-7B",
         "1.50 / 3.00", "50.0%",
         "7  (7%)", "54  (54%)", "21  (21%)", "18  (18%)"],
    ]
    story.append(tbl(overall_data,
                     [3.5*cm, 2.4*cm, 1.9*cm, 2.5*cm, 2.1*cm, 2.5*cm, 1.8*cm],
                     [("ALIGN", (1, 0), (-1, -1), "CENTER"),
                      ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold")]))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 9.3 Per-Category Breakdown
    # -----------------------------------------------------------------------
    story.append(Paragraph("9.3 Per-Category Breakdown", H2))
    story.append(Paragraph(
        "The table below shows average scores per knowledge category. Qwen leads in every single "
        "category. The largest gaps are in vLLM Flags &amp; Configuration (+0.87) and "
        "Replicas vs Tensor Parallelism (+0.90) — areas where Mistral most frequently answered "
        "from the wrong domain context.", BODY))

    cat_data = [["Category", "Qs", "Qwen", "Mistral", "Qwen Lead"]]
    for name, n, q, m in R["categories"]:
        lead = f"+{q - m:.2f}"
        cat_data.append([name, str(n), f"{q:.2f}", f"{m:.2f}", lead])
    cat_data.append(["Overall", "100", "2.17", "1.50", "+0.67"])

    n_rows = len(cat_data)
    story.append(tbl(cat_data,
                     [6.0*cm, 1.5*cm, 2.0*cm, 2.2*cm, 2.0*cm],
                     [("ALIGN", (1, 0), (-1, -1), "CENTER"),
                      ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                      ("FONTNAME", (0, n_rows-1), (-1, n_rows-1), "Helvetica-Bold"),
                      ("BACKGROUND", (0, n_rows-1), (-1, n_rows-1), LGRAY),
                      ("TEXTCOLOR",  (4, 1), (4, -1), GREEN_TXT),
                      ("FONTNAME",   (4, 1), (4, -1), "Helvetica-Bold")]))
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # 9.4 Key Qualitative Findings
    # -----------------------------------------------------------------------
    story.append(Paragraph("9.4 Key Qualitative Findings", H2))

    findings = [
        ("<b>Qwen strongest on KV Cache (2.67/3.00).</b> Qwen demonstrated near-complete mastery of "
         "KV cache mechanics — memory allocation, PagedAttention, GQA, eviction, prefix caching, "
         "and bandwidth saturation. 11 of 15 questions scored 3 (fully correct)."),

        ("<b>Mistral’s dominant failure mode: wrong domain context.</b> Mistral answered 18 "
         "questions with a score of 0, frequently substituting entirely wrong domains — defining "
         "ITL as ‘In-Memory Table Limit in Apache Flink’, TTFT as ‘Time to First Text’ "
         "in web performance, and GQA as a key-value store query optimisation technique. This pattern "
         "indicates Mistral-7B is not reliably grounded in vLLM-specific inference engineering knowledge."),

        ("<b>Largest Qwen advantage: vLLM flags and parallelism (Section 61–85).</b> On vLLM "
         "Flags &amp; Configuration, Qwen scored 2.20 vs Mistral’s 1.33. Mistral repeatedly confused "
         "vLLM flags with TensorFlow or generic PyTorch parameters. On Replicas vs Tensor Parallelism, "
         "Qwen scored 2.10 vs Mistral’s 1.20. Both categories are directly relevant to a "
         "production inference engineering tutor."),

        ("<b>Both models weakest on Chunked Prefill &amp; Prefix Caching (86–100).</b> Qwen scored "
         "1.67 and Mistral 1.00 on this category — the most technically advanced section. Both "
         "models incorrectly stated that prefix caching does not work across different users (Q90), "
         "and neither correctly explained the online softmax / Flash Attention relationship (Q99). "
         "This is the area requiring the most improvement in both models."),

        ("<b>Score distributions reveal different error profiles.</b> Qwen’s errors skew toward "
         "partial credit (42 scores of 2) — mostly right with missing detail. Mistral’s "
         "errors skew toward zero (18 completely wrong) — fundamental misunderstandings. For a "
         "production tutor, Mistral’s failure mode (confidently wrong answer in a different domain) "
         "is more harmful than Qwen’s (correct concept, incomplete explanation)."),
    ]

    for f in findings:
        story.append(Paragraph(f"•  {f}", BULLET))
    story.append(Spacer(1, 6))

    # -----------------------------------------------------------------------
    # 9.5 Implications for Deployment Recommendation
    # -----------------------------------------------------------------------
    story.append(Paragraph("9.5 Implications for Deployment Recommendation", H2))
    story.append(Paragraph(
        "The quality evaluation resolves the open question from Section 6.4. There is no "
        "quality-vs-latency trade-off to make: Qwen3-VL-4B wins on both dimensions. "
        "The table below summarises the complete evidence base for the deployment decision.", BODY))

    combined_data = [
        ["Dimension", "Qwen3-VL-4B", "Mistral-7B", "Winner"],
        # Serving
        ["Serving: Composite score at 100u",  "67.8M",              "49.0M",    "Qwen +38%"],
        ["Serving: ITL p95 stability",         "5.6–7.1ms (stable)", "Spikes to 29.6ms at 150u", "Qwen ✓"],
        ["Serving: Peak throughput",           "4,528 chunks/s",    "3,960 chunks/s", "Qwen +14%"],
        # Quality
        ["Quality: Overall avg score (0–3)", "2.17 / 3.00 (72%)", "1.50 / 3.00 (50%)", "Qwen +45%"],
        ["Quality: Fully correct answers",     "40 / 100 (40%)",    "7 / 100 (7%)",  "Qwen ✓"],
        ["Quality: Hallucination rate (0s)",   "5 / 100 (5%)",      "18 / 100 (18%)", "Qwen ✓"],
        ["Quality: Best category (KV Cache)",  "2.67 / 3.00",       "2.27 / 3.00",   "Qwen +0.40"],
        ["Quality: Worst category (Chunked)",  "1.67 / 3.00",       "1.00 / 3.00",   "Qwen +0.67"],
    ]
    n_comb = len(combined_data)
    story.append(tbl(combined_data,
                     [5.8*cm, 3.5*cm, 3.5*cm, 2.5*cm],
                     [("ALIGN",    (1, 0), (-1, -1), "CENTER"),
                      ("FONTNAME", (0, 1), (0, -1),  "Helvetica-Bold"),
                      ("TEXTCOLOR",(3, 1), (3, -1),  GREEN_TXT),
                      ("FONTNAME", (3, 1), (3, -1),  "Helvetica-Bold"),
                      # light separator between serving and quality rows
                      ("LINEABOVE",(0, 4), (-1, 4), 1.0, GRAY)]))
    story.append(Spacer(1, 12))

    # Final recommendation box
    rec_para = Paragraph(
        "<b>Confirmed recommendation: Deploy Qwen3-VL-4B at 100 concurrent users per H100.</b> "
        "This configuration delivers the highest composite serving score (67.8M), stable ITL "
        "(5.6 ms p95), and the highest domain knowledge accuracy (2.17/3.00). "
        "Qwen3-VL-4B is the production deployment choice on every measured dimension.",
        ParagraphStyle("RecBox", parent=BODY,
                       backColor=LBLUE_BG,
                       borderColor=LBLUE_BD, borderWidth=1, borderPadding=8,
                       spaceAfter=0))
    story.append(rec_para)

    doc.build(story)
    buf.seek(0)
    return buf


# ===========================================================================
# 5. ASSEMBLE final PDF
# ===========================================================================
def main():
    if not os.path.exists(INPUT_PDF):
        print(f"ERROR: Input PDF not found at:\n  {INPUT_PDF}")
        sys.exit(1)

    print(f"Input:  {INPUT_PDF}")
    print(f"Output: {OUTPUT_PDF}")
    print()

    # --- Probe positions ---
    print("Probing page element positions...")
    pos = probe_positions(INPUT_PDF)
    print(f"  Page 13 caveat box top   : {pos['caveat_top']:.1f} pt from page top")
    print(f"  Page 15 conclusion para  : {pos['conclusion_para_top']:.1f} pt from page top")
    print(f"  Page 15 resume box top   : {pos['resume_top']:.1f} pt from page top")

    # --- Build overlays ---
    print("Building page 13 overlay (caveat update)...")
    overlay_p13 = make_page13_overlay(pos["caveat_top"])

    print("Building page 15 overlay (conclusion update)...")
    overlay_p15 = make_page15_overlay(
        pos["conclusion_para_top"], pos["resume_top"]
    )

    # --- Build Section 9 ---
    print("Building Section 9 pages...")
    s9_buf = make_section9()
    s9_reader = PdfReader(s9_buf)

    # --- Assemble ---
    print("Assembling final PDF...")
    reader = PdfReader(INPUT_PDF)
    writer = PdfWriter()

    # Pages 1–12: unchanged
    for i in range(12):
        writer.add_page(reader.pages[i])

    # Page 13: merge overlay
    p13 = reader.pages[12]
    p13.merge_page(overlay_p13)
    writer.add_page(p13)

    # Page 14: unchanged (Section 7 Lessons Learned)
    writer.add_page(reader.pages[13])

    # Page 15: merge overlay
    p15 = reader.pages[14]
    p15.merge_page(overlay_p15)
    writer.add_page(p15)

    # Section 9 pages
    for page in s9_reader.pages:
        writer.add_page(page)

    with open(OUTPUT_PDF, "wb") as f:
        writer.write(f)

    total = len(writer.pages)
    print()
    print(f"Done! {total} pages written to:")
    print(f"  {OUTPUT_PDF}")
    print()
    print("Summary of changes:")
    print(f"  Page 13  — caveat box updated (references Section 9)")
    print(f"  Page 15  — conclusion updated (quality eval paragraph + resume box)")
    print(f"  Pages 16–{total} — Section 9: Answer Quality Evaluation (new)")


if __name__ == "__main__":
    main()
