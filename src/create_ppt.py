"""Generate a concise, modern 6-slide PowerPoint presentation for SmartPulse AI."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(15, 41, 66)
BLUE = RGBColor(37, 99, 235)
CYAN = RGBColor(0, 153, 204)
SLATE = RGBColor(51, 65, 85)
LIGHT_BG = RGBColor(244, 247, 250)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(22, 163, 74)


def _add_header(slide, title_text: str, subtitle_text: str = "") -> None:
    """Add a consistent top banner bar to content slides."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.05), Inches(13.333), Inches(0.06))
    accent.fill.solid()
    accent.fill.fore_color.rgb = CYAN
    accent.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.18), Inches(12.1), Inches(0.75))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = WHITE
    if subtitle_text:
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.size = Pt(12)
        p2.font.color.rgb = RGBColor(186, 230, 253)


def _add_bullets(slide, left: float, top: float, width: float, height: float, items: list[tuple[str, str]], font_size: int = 15) -> None:
    """Add clean bold-lead bullet points inside a light card."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT_BG
    card.line.color.rgb = RGBColor(203, 213, 225)

    tb = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.2), Inches(width - 0.5), Inches(height - 0.4))
    tf = tb.text_frame
    tf.word_wrap = True

    for idx, (head, body) in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        r1 = p.add_run()
        r1.text = f"• {head}: "
        r1.font.bold = True
        r1.font.size = Pt(font_size)
        r1.font.color.rgb = NAVY

        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(font_size)
        r2.font.color.rgb = SLATE


def build_presentation(project_root: Path) -> Path:
    """Create the 6-slide SmartPulse_AI_Presentation.pptx deck."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # --- Slide 1: Title ---
    s1 = prs.slides.add_slide(blank_layout)
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(5.75), Inches(13.333), Inches(0.08))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = CYAN
    stripe.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(3.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = "SmartPulse AI"
    p1.font.size = Pt(48)
    p1.font.bold = True
    p1.font.color.rgb = CYAN

    p2 = tf.add_paragraph()
    p2.text = "Congestion-Aware Emergency Ambulance Router"
    p2.font.size = Pt(30)
    p2.font.bold = True
    p2.font.color.rgb = WHITE
    p2.space_before = Pt(8)

    p3 = tf.add_paragraph()
    p3.text = "AI Lab Project  |  K-Nearest Neighbors (KNN) + A* Informed Search vs. BFS"
    p3.font.size = Pt(17)
    p3.font.color.rgb = RGBColor(203, 213, 225)
    p3.space_before = Pt(18)

    footer = s1.shapes.add_textbox(Inches(1.0), Inches(6.05), Inches(11.3), Inches(0.8))
    fp = footer.text_frame.paragraphs[0]
    fp.text = "Dataset: Kaggle Traffic Prediction (5,952 records)   •   github.com/hamzahk01"
    fp.font.size = Pt(14)
    fp.font.color.rgb = RGBColor(148, 163, 184)

    # --- Slide 2: Problem & Pipeline ---
    s2 = prs.slides.add_slide(blank_layout)
    _add_header(s2, "Problem & System Pipeline", "Bridging Machine Learning and Informed Pathfinding")
    _add_bullets(
        s2,
        0.6,
        1.45,
        12.13,
        2.45,
        [
            ("Problem", "Shortest-hop routing (BFS) ignores rush-hour congestion, delaying emergency response."),
            ("Unique Solution", "Train KNN on Kaggle traffic sensor data to predict dynamic road delays, then route with A*."),
            ("Mission", "2-leg dispatch on a 16x16 urban grid: Depot (1,1) -> Patient (8,11) -> Hospital (14,14)."),
        ],
        font_size=16,
    )

    steps = [
        ("1. Kaggle Data", "TrafficTwoMonth.csv\n5,952 records"),
        ("2. KNN Model", "StandardScaler + k=9\n91.60% Test Acc"),
        ("3. City Cost Grid", "Low: 1m | Normal: 2m\nHigh: 4m | Heavy: 7.5m"),
        ("4. A* Search", "f(n) = g(n) + h(n)\n34.3% faster vs BFS"),
    ]
    for i, (st_title, st_body) in enumerate(steps):
        x = 0.6 + i * 3.12
        box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.35), Inches(2.85), Inches(2.2))
        box.fill.solid()
        box.fill.fore_color.rgb = BLUE if i < 3 else GREEN
        box.line.fill.background()

        bt = s2.shapes.add_textbox(Inches(x + 0.15), Inches(4.6), Inches(2.55), Inches(1.7))
        btf = bt.text_frame
        btf.word_wrap = True
        bp1 = btf.paragraphs[0]
        bp1.alignment = PP_ALIGN.CENTER
        bp1.text = st_title
        bp1.font.size = Pt(17)
        bp1.font.bold = True
        bp1.font.color.rgb = WHITE

        bp2 = btf.add_paragraph()
        bp2.alignment = PP_ALIGN.CENTER
        bp2.space_before = Pt(12)
        bp2.text = st_body
        bp2.font.size = Pt(13)
        bp2.font.color.rgb = RGBColor(226, 232, 240)

    # --- Slide 3: KNN Model ---
    s3 = prs.slides.add_slide(blank_layout)
    _add_header(s3, "ML Component: K-Nearest Neighbors (KNN)", "Trained on Kaggle Traffic Prediction Dataset")
    _add_bullets(
        s3,
        0.6,
        1.45,
        4.6,
        5.3,
        [
            ("Features (8)", "TimeSin, TimeCos, IsWeekend, Car, Bike, Bus, Truck, HeavyRatio."),
            ("Tuning", "5-Fold CV tested odd k in [1..21]; optimal k = 9 (92.82% CV)."),
            ("Test Accuracy", "91.60% on untouched 20% holdout (Macro F1: 0.8668)."),
            ("Classes", "Predicts low, normal, high, heavy congestion."),
        ],
        font_size=14,
    )
    knn_img = project_root / "outputs" / "knn_evaluation.png"
    if knn_img.exists():
        s3.shapes.add_picture(str(knn_img), Inches(5.45), Inches(1.7), width=Inches(7.35))

    # --- Slide 4: Search Comparison Visual ---
    s4 = prs.slides.add_slide(blank_layout)
    _add_header(s4, "Search Algorithms: A* vs. BFS Baseline", "Optimal Minimum-Delay Path vs. Blind Shortest-Hop Path")
    _add_bullets(
        s4,
        0.6,
        1.35,
        12.13,
        1.4,
        [
            ("A* Search", "Minimizes f(n) = g(n) + h(n) with admissible Manhattan heuristic (h = dist * 1.0 min) — guaranteed optimal."),
            ("BFS Baseline", "Unweighted FIFO search finds fewest hops (26) but cuts straight through heavy downtown traffic."),
        ],
        font_size=14,
    )
    route_img = project_root / "outputs" / "route_comparison.png"
    if route_img.exists():
        s4.shapes.add_picture(str(route_img), Inches(0.6), Inches(2.95), width=Inches(12.13))

    # --- Slide 5: Benchmark Summary ---
    s5 = prs.slides.add_slide(blank_layout)
    _add_header(s5, "Experimental Results", "Quantitative Comparison Across Dispatch Scenarios")

    rows = [
        ["Scenario", "Algorithm", "Travel Time", "Hops", "Nodes Explored", "Time Saved"],
        ["Morning Rush (Mon 08:00)", "A* + KNN", "69.0 min", "32", "188", "36.0 min (34.3%)"],
        ["Morning Rush (Mon 08:00)", "BFS Baseline", "105.0 min", "26", "281", "—"],
        ["Late Night (Sun 02:00)", "A* + KNN", "45.0 min", "26", "154", "4.0 min (8.2%)"],
        ["Late Night (Sun 02:00)", "BFS Baseline", "49.0 min", "26", "281", "—"],
    ]
    tbl_shape = s5.shapes.add_table(len(rows), 6, Inches(0.6), Inches(1.6), Inches(12.13), Inches(3.4))
    tbl = tbl_shape.table
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            p.font.size = Pt(14)
            if r_idx == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
                p.font.bold = True
                p.font.color.rgb = WHITE
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG if r_idx % 2 == 1 else WHITE
                p.font.bold = (c_idx == 5 and "34.3%" in val)
                p.font.color.rgb = GREEN if (c_idx == 5 and "%" in val) else SLATE

    _add_bullets(
        s5,
        0.6,
        5.35,
        12.13,
        1.55,
        [
            ("Key Takeaway", "A* + KNN reduces rush-hour travel delay by 34.3% and explores 33%–45% fewer nodes than BFS."),
        ],
        font_size=15,
    )

    # --- Slide 6: How to Run ---
    s6 = prs.slides.add_slide(blank_layout)
    _add_header(s6, "How to Run & Deliverables", "Reproducible Single-Command Pipeline")
    _add_bullets(
        s6,
        0.6,
        1.55,
        12.13,
        5.1,
        [
            ("Run Full Pipeline", "python main.py   (Downloads Kaggle CSV, trains KNN, routes A* & BFS, builds PDF)"),
            ("Custom Scenario", "python main.py --hour 18 --day Friday"),
            ("Run Unit Tests", "python -m unittest discover -s tests -v   (4/4 passing)"),
            ("Outputs", "models/knn_traffic_model.pkl, outputs/*.png, SmartPulse_AI_Project_Report.pdf"),
        ],
        font_size=16,
    )

    out_path = project_root / "SmartPulse_AI_Presentation.pptx"
    prs.save(str(out_path))
    print(f"[PPT] Generated presentation at: {out_path}")
    return out_path


if __name__ == "__main__":
    build_presentation(Path(__file__).resolve().parent.parent)
