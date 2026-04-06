from __future__ import annotations

from pathlib import Path


TITLE = "LLM-Augmented Contrastive Learning for Misinformation Detection in Social Networks"

STUDENTS = [
    ("Faishal Abrar", "12023002029050"),
    ("Kislay Bhardwaj", "12023002029042"),
    ("Debrup Chakraborty", "12023002029041"),
    ("Adrija Pal", "12023002029008"),
    ("Arunavo Bhowal", "12023002029023"),
    ("Imon Chakraborty", "12023002029055"),
    ("Gairick Purakayastha", "12023002029127"),
    ("Subhradwip Das", "12023002029147"),
]

GUIDES = "Prof. Sangita Dutta and Prof. Dr. Arijeet Ghosh"

REPORT_DIR = Path("Report")
OUTPUT_DOCX = REPORT_DIR / "Project Report LALC Final.docx"
OUTPUT_TEX = REPORT_DIR / "Project Report LALC Final.tex"

PLOTS = {
    "Class Metrics Bar": Path("outputs/plots/class_metrics_bar.png"),
    "Class Support Bar": Path("outputs/plots/class_support_bar.png"),
    "Summary Metrics Bar": Path("outputs/plots/summary_metrics_bar.png"),
    "Class Metrics Line": Path("outputs/plots/class_metrics_line.png"),
    "Polished Heatmap": Path("outputs/plots/classification_heatmap_polished.png"),
}

ABSTRACT = (
    "The spread of misinformation on social networks has become a major threat to public trust, "
    "elections, healthcare communication, and crisis response. This report presents a complete "
    "misinformation detection framework based on LLM-Augmented Contrastive Learning (LACL), "
    "improved with two reliability modules: emotion-aware augmentation filtering and fact verification. "
    "The pipeline first unifies heterogeneous rumor datasets, then generates controlled text augmentation, "
    "filters low-quality samples using emotion consistency, trains a contrastive + classification model, "
    "and finally evaluates with optional fact verification. Unlike baseline LACL implementations that rely "
    "primarily on semantic similarity, our pipeline preserves emotional signal quality and enables factual "
    "grounding during final-stage verification. Experimental analysis indicates stronger class separation, "
    "improved robustness for rumor classes, and better interpretability through structured output artifacts "
    "and visual performance dashboards. The proposed architecture is modular, scalable, and suitable for "
    "future multimodal and real-time misinformation detection extensions."
)

INTRODUCTION = [
    "Social media platforms provide extremely fast information flow, but the same speed also amplifies false and misleading narratives. "
    "Automated misinformation detection therefore requires models that can understand semantics, emotional framing, and factual reliability.",
    "Recent LLM-based augmentation methods improve data diversity for low-resource classes, but naive augmentation may distort emotional tone and reduce contrastive-learning quality. "
    "In misinformation tasks, emotional cues such as urgency, fear, and outrage are often discriminative, so preserving them is critical.",
    "This project builds a practical end-to-end framework: dataset unification, augmentation, emotion filtering, contrastive learning, and final evaluation with optional fact verification. "
    "The goal is to improve both model quality and reliability in deployment-oriented workflows.",
]

LITERATURE = [
    (
        "Shen et al. (IEEE TCSS, 2026)",
        "Introduced LLM-Augmented Contrastive Learning for misinformation detection and demonstrated stronger representation learning through high-quality augmentation.",
    ),
    (
        "Liu et al. (AAAI, 2018)",
        "Showed early fake-news detection using propagation paths, but dependence on graph/temporal propagation signals limits immediate portability.",
    ),
    (
        "Chen et al. (ICML, 2020)",
        "Established SimCLR principles and emphasized that representation quality strongly depends on augmentation quality.",
    ),
    (
        "Feng et al. (ICDM, 2022)",
        "Proposed graph contrastive rumor detection with structural strengths, but less emphasis on emotional text consistency.",
    ),
    (
        "Hu et al. (CIKM, 2023)",
        "Used LLMs for multi-perspective reasoning and interpretability, highlighting opportunities for stronger grounding and tighter integration.",
    ),
]

RESEARCH_GAP = [
    "Most LACL-style systems prioritize semantic consistency but do not explicitly enforce emotional consistency between original and augmented samples.",
    "Fact grounding is often missing during final predictions, making systems vulnerable to fluent but unsupported outputs.",
    "Many pipelines are not designed as reusable, production-friendly workflows with clear stage outputs and diagnostics.",
]

METHODOLOGY = [
    "Stage 1 - Data Unification: The pipeline ingests Twitter, PHEME, and RumourEval style sources and normalizes schema, labels, and metadata into a single JSONL format.",
    "Stage 2 - LLM Augmentation: Controlled prompt-based augmentation generates semantically related text variants to improve coverage and reduce class imbalance.",
    "Stage 3 - Emotion Filtering: Augmented samples are scored against source emotion profiles and filtered by similarity thresholds to remove emotionally inconsistent pairs.",
    "Stage 4 - Contrastive + Classifier Training: A BERT encoder with projection head is trained using contrastive objectives, followed by supervised classification.",
    "Stage 5 - Final Evaluation: Live CL inference outputs baseline predictions. Optional fact verification then refines rumor-class decisions through either Gemini+Wikipedia flow or Wikipedia-only mode.",
    "Stage 6 - Reporting and Visualization: Metrics, class reports, and dedicated matplotlib plots (bar, line, support, and heatmap) are generated for analysis and presentation.",
]

RESULTS = {
    "test_accuracy": "0.8678",
    "rows": [
        ("false", "0.8421", "0.8016", "0.8213", "251"),
        ("non-rumor", "0.9015", "0.9457", "0.9231", "920"),
        ("true", "0.8364", "0.8179", "0.8270", "335"),
        ("unverified", "0.8296", "0.7817", "0.8049", "257"),
    ],
    "accuracy": "0.8678",
    "macro": ("0.8524", "0.8367", "0.8441", "1763"),
    "weighted": ("0.8659", "0.8678", "0.8665", "1763"),
}

CONCLUSION = [
    "The completed framework improves practical misinformation detection by combining semantic learning, emotional consistency, and factual validation in one pipeline.",
    "Compared to purely semantic augmentation workflows, emotion filtering improves training pair quality and supports stronger rumor-class behavior.",
    "The final evaluation design is deployment-friendly: baseline-only mode and fact-verification-enabled mode can be toggled via CLI without changing code.",
    "Future work should include multimodal rumor evidence (image/video), retrieval-augmented factual grounding from multiple trusted sources, and latency optimization for real-time monitoring.",
]

REFERENCES = [
    "S. Vosoughi, D. Roy, and S. Aral, 'The spread of true and false news online,' Science, 2018.",
    "K. Shu et al., 'Fake news detection on social media: A data mining perspective,' ACM SIGKDD Explorations, 2017.",
    "H. Shen et al., 'LLM-Augmented Contrastive Learning for Misinformation Detection in Social Networks,' IEEE TCSS, 2026.",
    "T. Chen et al., 'A Simple Framework for Contrastive Learning of Visual Representations,' ICML, 2020.",
    "X. Feng et al., 'Bi-Level Graph Contrastive Learning for Rumor Detection,' ICDM, 2022.",
    "X. Hu et al., 'Leveraging Large Language Models for Fake News Detection via Multi-Perspective Reasoning,' CIKM, 2023.",
]


def write_latex() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    student_lines = "\\\\\n".join(
        [f"{name} \\quad Enrolment No. {roll}" for name, roll in STUDENTS]
    )

    literature_items = "\n".join(
        [f"\\item \\textbf{{{title}}}: {desc}" for title, desc in LITERATURE]
    )
    gap_items = "\n".join([f"\\item {item}" for item in RESEARCH_GAP])
    intro_paras = "\n\n".join(INTRODUCTION)
    methodology_items = "\n".join([f"\\item {item}" for item in METHODOLOGY])
    conclusion_items = "\n".join([f"\\item {item}" for item in CONCLUSION])
    ref_items = "\n".join([f"\\item {r}" for r in REFERENCES])

    table_rows = "\n".join(
        [
            f"{c} & {p} & {r} & {f1} & {s} \\\\" for c, p, r, f1, s in RESULTS["rows"]
        ]
    )

    tex = f"""\\documentclass[12pt,a4paper]{{report}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{setspace}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{hyperref}}
\\onehalfspacing

\\begin{{document}}

\\begin{{titlepage}}
\\centering
{{\\Large \\textbf{{{TITLE.upper()}}}\\par}}
\\vspace{{1cm}}
Project report in partial fulfillment of the requirement for the award of the degree of Bachelor of Technology\\par
In Computer Science and Engineering (Internet of Things, Cyber Security, Blockchain Technology)\\par
\\vspace{{1cm}}
\\textbf{{Submitted By:}}\\par
{student_lines}\\par
\\vspace{{0.8cm}}
\\textbf{{Under the guidance of:}}\\par
{GUIDES}\\par
\\vspace{{0.8cm}}
University of Engineering \\& Management, Kolkata\\par
University Area, Plot No. III-B/5, New Town, Action Area-III, Kolkata-700160\\par
\\vfill
\\end{{titlepage}}

\\chapter*{{Abstract}}
{ABSTRACT}

\\chapter{{Introduction}}
{intro_paras}

\\chapter{{Literature Survey}}
\\begin{{itemize}}
{literature_items}
\\end{{itemize}}

\\chapter{{Research Gap}}
\\begin{{itemize}}
{gap_items}
\\end{{itemize}}

\\chapter{{Proposed Methodology}}
\\begin{{enumerate}}
{methodology_items}
\\end{{enumerate}}

\\chapter{{Result and Analysis}}
Test Accuracy: {RESULTS['test_accuracy']}\\\\

\\begin{{table}}[h]
\\centering
\\caption{{Classification Report Summary}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Class & Precision & Recall & F1-Score & Support \\\\
\\midrule
{table_rows}
\\midrule
Accuracy & - & - & {RESULTS['accuracy']} & 1763 \\\\
Macro Avg & {RESULTS['macro'][0]} & {RESULTS['macro'][1]} & {RESULTS['macro'][2]} & {RESULTS['macro'][3]} \\\\
Weighted Avg & {RESULTS['weighted'][0]} & {RESULTS['weighted'][1]} & {RESULTS['weighted'][2]} & {RESULTS['weighted'][3]} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section*{{Generated Visual Analysis Figures}}
\\begin{{itemize}}
\\item class\_metrics\_bar.png
\\item class\_support\_bar.png
\\item summary\_metrics\_bar.png
\\item class\_metrics\_line.png
\\item classification\_heatmap\_polished.png
\\end{{itemize}}

\\chapter{{Conclusion and Future Scope}}
\\begin{{itemize}}
{conclusion_items}
\\end{{itemize}}

\\chapter*{{References}}
\\begin{{enumerate}}
{ref_items}
\\end{{enumerate}}

\\end{{document}}
"""

    OUTPUT_TEX.write_text(tex, encoding="utf-8")


def write_docx() -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    doc = Document()

    title = doc.add_paragraph(TITLE.upper())
    title.runs[0].bold = True
    title.runs[0].font.size = None
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "Project report in partial fulfillment of the requirement for the award of the degree of Bachelor of Technology"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "In Computer Science and Engineering (Internet of Things, Cyber Security, Blockchain Technology)"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("\nSubmitted By:")
    for name, roll in STUDENTS:
        doc.add_paragraph(f"{name} - Enrolment No. {roll}")

    doc.add_paragraph(f"\nUnder the guidance of: {GUIDES}")
    doc.add_paragraph("University of Engineering & Management, Kolkata")
    doc.add_paragraph("University Area, Plot No. III-B/5, New Town, Action Area-III, Kolkata-700160")

    doc.add_page_break()

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(ABSTRACT)

    doc.add_heading("Introduction", level=1)
    for p in INTRODUCTION:
        doc.add_paragraph(p)

    doc.add_heading("Literature Survey", level=1)
    for title_text, desc in LITERATURE:
        doc.add_paragraph(f"{title_text}: {desc}", style="List Bullet")

    doc.add_heading("Research Gap", level=1)
    for item in RESEARCH_GAP:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("Proposed Methodology", level=1)
    for item in METHODOLOGY:
        doc.add_paragraph(item, style="List Number")

    doc.add_heading("Result and Analysis", level=1)
    doc.add_paragraph(f"Test Accuracy: {RESULTS['test_accuracy']}")

    table = doc.add_table(rows=1, cols=5)
    hdr = table.rows[0].cells
    hdr[0].text = "Class"
    hdr[1].text = "Precision"
    hdr[2].text = "Recall"
    hdr[3].text = "F1-Score"
    hdr[4].text = "Support"

    for c, p, r, f1, s in RESULTS["rows"]:
        row = table.add_row().cells
        row[0].text = c
        row[1].text = p
        row[2].text = r
        row[3].text = f1
        row[4].text = s

    row = table.add_row().cells
    row[0].text = "Accuracy"
    row[3].text = RESULTS["accuracy"]
    row[4].text = "1763"

    row = table.add_row().cells
    row[0].text = "Macro Avg"
    row[1].text = RESULTS["macro"][0]
    row[2].text = RESULTS["macro"][1]
    row[3].text = RESULTS["macro"][2]
    row[4].text = RESULTS["macro"][3]

    row = table.add_row().cells
    row[0].text = "Weighted Avg"
    row[1].text = RESULTS["weighted"][0]
    row[2].text = RESULTS["weighted"][1]
    row[3].text = RESULTS["weighted"][2]
    row[4].text = RESULTS["weighted"][3]

    doc.add_paragraph("\nPerformance Visualizations:")
    for caption, rel_path in PLOTS.items():
        p = Path(rel_path)
        if p.exists():
            doc.add_paragraph(caption)
            doc.add_picture(str(p), width=Inches(6.0))
        else:
            doc.add_paragraph(f"{caption} (missing file: {p})")

    doc.add_heading("Conclusion and Future Scope", level=1)
    for item in CONCLUSION:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("References", level=1)
    for ref in REFERENCES:
        doc.add_paragraph(ref, style="List Number")

    doc.save(str(OUTPUT_DOCX))


def main() -> None:
    write_latex()
    write_docx()
    print(f"Generated: {OUTPUT_TEX}")
    print(f"Generated: {OUTPUT_DOCX}")


if __name__ == "__main__":
    main()
