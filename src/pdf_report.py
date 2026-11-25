"""Generate a comprehensive PDF project report including metrics, visuals, and code explanations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.city_router import CONGESTION_COSTS, DEPOT, GRID_SIZE, HOSPITAL, PATIENT
from src.download_data import KAGGLE_DATASET_SLUG
from src.knn_model import FEATURE_COLUMNS, TRAFFIC_CLASSES


def _styled_table(data: list[list[Any]], col_widths: list[float] | None = None) -> Table:
    """Create a consistently styled ReportLab table."""
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def generate_pdf_report(
    pdf_path: Path,
    metrics: dict[str, Any],
    scenarios: list[dict[str, Any]],
    knn_plot_path: Path,
    route_plot_path: Path,
) -> Path:
    """Build the complete SmartPulse AI Lab project report PDF."""
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="SmartPulse AI Lab Project Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f2942"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1b365d"),
        spaceBefore=12,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#2563eb"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9.8,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        "CodeBlock",
        parent=styles["Code"],
        fontSize=8.3,
        leading=11.5,
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8,
    )

    story: list[Any] = []

    # --- Header ---
    story.append(Paragraph("SmartPulse AI: Congestion-Aware Emergency Ambulance Router", title_style))
    story.append(
        Paragraph(
            f"AI Lab Project Report — Combining K-Nearest Neighbors (KNN) &amp; A* Informed Search "
            f"(with BFS Baseline) | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            subtitle_style,
        )
    )

    # --- Section 1: Overview ---
    story.append(Paragraph("1. Project Overview &amp; Unique Concept", h1))
    story.append(
        Paragraph(
            "Traditional GPS shortest-path algorithms (such as unweighted Breadth-First Search) optimize only "
            "for physical distance or hop count. During urban emergencies, routing an ambulance down the "
            "shortest downtown avenue during rush hour can trap the vehicle in severe gridlock. "
            "<b>SmartPulse AI</b> solves this by tightly coupling a supervised Machine Learning model "
            "(<b>K-Nearest Neighbors</b>) with an informed graph search algorithm (<b>A* Search</b>).",
            body,
        )
    )
    story.append(
        Paragraph(
            "First, a KNN classifier is trained on real traffic sensor observations downloaded directly from "
            f"<b>Kaggle</b> (<i>{KAGGLE_DATASET_SLUG}</i>) to classify road segments into four congestion levels: "
            "<b>low</b>, <b>normal</b>, <b>high</b>, and <b>heavy</b>. Next, a 16x16 urban road network is "
            "populated with held-out Kaggle sensor readings for a chosen dispatch time (e.g., Monday 08:30 AM "
            "Rush Hour vs. Sunday 02:30 AM Off-Peak). The trained KNN model dynamically predicts edge traversal "
            "delays across every road cell, and A* Search computes the fastest two-leg rescue route "
            f"(<b>Depot {DEPOT} -&gt; Patient {PATIENT} -&gt; Hospital {HOSPITAL}</b>), outperforming BFS.",
            body,
        )
    )

    # --- Section 2: Dataset ---
    story.append(Paragraph("2. Kaggle Dataset &amp; Feature Engineering", h1))
    story.append(
        Paragraph(
            f"The dataset (<b>TrafficTwoMonth.csv</b>, {metrics['dataset_rows']} records) contains 15-minute "
            "interval traffic counts captured via computer vision sensors: <i>Time, Date, Day of the week, "
            "CarCount, BikeCount, BusCount, TruckCount, Total,</i> and target label <i>Traffic Situation</i>. "
            "An 80/20 stratified split reserves "
            f"<b>{metrics['train_rows']}</b> training samples and <b>{metrics['test_rows']}</b> untouched test samples.",
            body,
        )
    )

    dist = metrics["class_distribution"]
    dist_table = [
        ["Traffic Class", "Total Records", "Assigned Travel Delay (min/cell)"],
        ["low", str(dist.get("low", 0)), f"{CONGESTION_COSTS['low']:.1f} min"],
        ["normal", str(dist.get("normal", 0)), f"{CONGESTION_COSTS['normal']:.1f} min"],
        ["high", str(dist.get("high", 0)), f"{CONGESTION_COSTS['high']:.1f} min"],
        ["heavy", str(dist.get("heavy", 0)), f"{CONGESTION_COSTS['heavy']:.1f} min"],
    ]
    story.append(_styled_table(dist_table, col_widths=[1.8 * inch, 1.8 * inch, 2.6 * inch]))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"<b>Engineered Features ({len(FEATURE_COLUMNS)} total):</b> "
            "Raw clock strings (e.g. <i>08:30:00 AM</i>) are converted into cyclic trigonometric coordinates "
            "<b>TimeSin</b> and <b>TimeCos</b> so 11:45 PM remains adjacent to 12:00 AM. We also compute "
            "<b>IsWeekend</b> (0/1), keep vehicle counts (<b>CarCount, BikeCount, BusCount, TruckCount</b>), "
            "and derive <b>HeavyRatio = (BusCount + TruckCount) / Total</b>. All features are normalized "
            "using <i>StandardScaler</i> before KNN distance calculations.",
            body,
        )
    )

    # --- Section 3: KNN Model ---
    story.append(Paragraph("3. Machine Learning Model: K-Nearest Neighbors (KNN)", h1))
    story.append(
        Paragraph(
            "KNN is a non-parametric, distance-based classifier that predicts the class of a query point "
            "by inspecting its <i>k</i> closest neighbors in standardized feature space using Euclidean distance "
            "<i>d(x, y) = sqrt(sum((x_i - y_i)^2))</i>. Using inverse-distance weighting "
            "(<i>weights='distance'</i>), closer historical traffic snapshots exert stronger voting influence. "
            "We evaluated odd values of <i>k</i> from 1 to 21 via 5-fold Stratified Cross-Validation on the "
            "training split alone.",
            body,
        )
    )

    summary_table = [
        ["Evaluation Metric", "Observed Result"],
        ["Optimal k (5-Fold CV)", str(metrics["best_k"])],
        ["Best 5-Fold CV Accuracy", f"{metrics['best_cv_accuracy'] * 100:.2f}%"],
        ["Held-Out Test Accuracy (20%)", f"{metrics['test_accuracy'] * 100:.2f}%"],
        ["Held-Out Macro F1-Score", f"{metrics['macro_f1']:.4f}"],
    ]
    story.append(_styled_table(summary_table, col_widths=[2.6 * inch, 3.6 * inch]))
    story.append(Spacer(1, 8))

    # Per-class report table
    rep = metrics["classification_report"]
    cls_rows = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
    for cls in TRAFFIC_CLASSES:
        c_info = rep.get(cls, {})
        cls_rows.append(
            [
                cls,
                f"{c_info.get('precision', 0.0):.3f}",
                f"{c_info.get('recall', 0.0):.3f}",
                f"{c_info.get('f1-score', 0.0):.3f}",
                str(int(c_info.get("support", 0))),
            ]
        )
    story.append(_styled_table(cls_rows, col_widths=[1.4 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]))
    story.append(Spacer(1, 10))

    if knn_plot_path.exists():
        story.append(Image(str(knn_plot_path), width=6.7 * inch, height=2.68 * inch))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # --- Section 4: Search Algorithms ---
    story.append(Paragraph("4. Informed Search (A*) vs. Uninformed Search (BFS)", h1))
    story.append(
        Paragraph(
            f"The city is modeled as a {GRID_SIZE}x{GRID_SIZE} grid containing impassable building blocks, "
            f"an Ambulance Depot at {DEPOT}, a Patient Pickup at {PATIENT}, and a Trauma Hospital at {HOSPITAL}. "
            "For each dispatch scenario, held-out Kaggle test observations matching that time-of-day and "
            "weekday/weekend profile are fed into the trained KNN model to assign live delay costs.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>A* Search:</b> Minimizes <i>f(n) = g(n) + h(n)</i> using a min-heap priority queue, where "
            "<i>g(n)</i> is cumulative KNN-predicted travel time in minutes and <i>h(n) = (|x1-x2| + |y1-y2|) * 1.0</i> "
            "is Manhattan distance scaled by the minimum possible per-cell delay (1.0 min). Because no step can "
            "cost less than 1.0 min, <i>h(n)</i> never overestimates the true remaining travel time — proving "
            "admissibility and guaranteeing A* returns the globally minimum-delay route.<br/>"
            "<b>BFS Baseline:</b> Uses a FIFO queue over unweighted edges, guaranteeing the fewest grid hops "
            "while ignoring congestion penalties.",
            body,
        )
    )

    scen_rows = [
        [
            "Scenario",
            "A* Time (min)",
            "BFS Time (min)",
            "A* Hops",
            "BFS Hops",
            "A* Nodes",
            "BFS Nodes",
            "Delay Saved",
        ]
    ]
    for s in scenarios:
        a = s["astar"]
        b = s["bfs"]
        scen_rows.append(
            [
                s["name"],
                f"{a.total_cost:.1f}",
                f"{b.total_cost:.1f}",
                str(a.total_hops),
                str(b.total_hops),
                str(a.nodes_explored),
                str(b.nodes_explored),
                f"{s['time_saved_min']:.1f}m ({s['time_saved_pct']:.1f}%)",
            ]
        )
    story.append(
        _styled_table(
            scen_rows,
            col_widths=[1.75 * inch, 0.72 * inch, 0.75 * inch, 0.58 * inch, 0.6 * inch, 0.62 * inch, 0.62 * inch, 1.06 * inch],
        )
    )
    story.append(Spacer(1, 10))

    if route_plot_path.exists():
        story.append(Image(str(route_plot_path), width=6.9 * inch, height=2.35 * inch))
        story.append(Spacer(1, 8))

    # --- Section 5: Complete Code Explanation ---
    story.append(PageBreak())
    story.append(Paragraph("5. Complete Code Explanation (Module-by-Module)", h1))

    modules = [
        (
            "5.1 src/download_data.py — Automated Kaggle Dataset Acquisition",
            "Defines <b>download_kaggle_dataset(data_dir, force, retries, timeout)</b>. "
            "It checks if <i>data/TrafficTwoMonth.csv</i> is already cached locally. If missing (or when "
            "<i>--force-download</i> is passed), it queries Kaggle's dataset download endpoint for "
            f"<i>{KAGGLE_DATASET_SLUG}</i> with exponential retry backoff, extracts the ZIP archive in memory "
            "using Python's <i>zipfile.ZipFile</i>, and writes <i>Traffic.csv</i> and <i>TrafficTwoMonth.csv</i> "
            "into <i>data/</i>.",
        ),
        (
            "5.2 src/knn_model.py — Feature Engineering, Tuning &amp; KNN Training",
            "<b>engineer_features(df)</b> parses 12-hour timestamps into radians and computes cyclic "
            "<i>TimeSin/TimeCos</i>, binary <i>IsWeekend</i>, vehicle counts, and <i>HeavyRatio</i>. "
            "<b>train_and_evaluate_knn()</b> splits data 80/20 with stratification, fits a <i>StandardScaler</i> "
            "exclusively on training data (preventing data leakage), loops over odd <i>k</i> in [1..21] using "
            "5-fold StratifiedKFold CV, trains the winning <i>KNeighborsClassifier(weights='distance')</i>, "
            "saves the model bundle + held-out test records to <i>models/knn_traffic_model.pkl</i>, and plots "
            "<i>outputs/knn_evaluation.png</i> via <b>plot_knn_evaluation()</b>.",
        ),
        (
            "5.3 src/city_router.py — City Grid, A* Search &amp; BFS Baseline",
            "<b>build_knn_traffic_grid()</b> filters the untouched Kaggle test records around the requested "
            "<i>hour</i> and <i>day_name</i>, assigns central commercial cells to busier samples and peripheral "
            "ring cells to lighter samples, and runs batch <b>KNN.predict()</b> to generate the congestion map. "
            "<b>astar_search()</b> maintains a <i>heapq</i> priority queue ordered by <i>f(n) = g(n) + h(n)</i> "
            "with admissible <b>_manhattan()</b> heuristic, while <b>bfs_search()</b> uses a <i>deque</i> for "
            "unweighted shortest-hop routing. <b>run_dispatch_scenario()</b> chains Depot -&gt; Patient -&gt; "
            "Hospital into a <b>SearchResult</b>, and <b>plot_route_comparison()</b> renders the maps and bar chart.",
        ),
        (
            "5.4 src/pdf_report.py — Automated Report Generation",
            "<b>generate_pdf_report()</b> uses ReportLab Platypus (<i>SimpleDocTemplate</i>, <i>Table</i>, "
            "<i>Paragraph</i>, <i>Image</i>) to compile live training metrics, classification tables, search "
            "comparisons, embedded PNG figures, code explanations, and usage instructions into this PDF.",
        ),
        (
            "5.5 main.py — CLI Orchestrator",
            "Parses optional flags (<i>--force-download</i>, <i>--hour</i>, <i>--day</i>, <i>--no-pdf</i>), "
            "executes Data Download -&gt; KNN Training -&gt; Rush Hour &amp; Off-Peak Routing (plus any custom "
            "scenario) -&gt; Visualization -&gt; PDF generation, and prints a clean summary dashboard.",
        ),
        (
            "5.6 tests/test_project.py — Automated Verification Tests",
            "Uses Python's <i>unittest</i> framework to verify that Kaggle data exists and loads cleanly, "
            "KNN test accuracy exceeds 85%, A* achieves total travel cost &lt;= BFS on Rush Hour, routes never "
            "enter building obstacles, and the PDF report is non-empty.",
        ),
    ]

    for title_txt, desc_txt in modules:
        story.append(Paragraph(title_txt, h2))
        story.append(Paragraph(desc_txt, body))

    story.append(Paragraph("Key Core Logic Snippet (A* Priority Update):", h2))
    story.append(
        Paragraph(
            "tentative_g = g_score[current] + float(cost_grid[nxt])<br/>"
            "if tentative_g &lt; g_score.get(nxt, float('inf')):<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;came_from[nxt] = current<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;g_score[nxt] = tentative_g<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;f_score = tentative_g + _manhattan(nxt, goal)<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;heapq.heappush(open_heap, (f_score, counter, nxt))",
            code_style,
        )
    )

    # --- Section 6: How to Run ---
    story.append(Paragraph("6. How to Run &amp; Experiment With the Project", h1))
    story.append(
        Paragraph(
            "Open PowerShell or Command Prompt and run the commands below inside the project directory:",
            body,
        )
    )
    story.append(
        Paragraph(
            "cd Desktop\\SmartPulse_AI_Lab<br/>"
            "pip install -r requirements.txt &nbsp;&nbsp;# Optional if packages are already installed<br/>"
            "python main.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Downloads Kaggle data, trains KNN, runs A* &amp; BFS, builds PDF<br/>"
            "python main.py --hour 18 --day Friday &nbsp;# Add a custom Friday 6 PM dispatch scenario<br/>"
            "python main.py --force-download &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Re-fetch fresh zip from kaggle.com<br/>"
            "python -m unittest discover -s tests -v &nbsp;# Run automated verification tests",
            code_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Generated Artifacts:</b><br/>"
            "- <i>data/TrafficTwoMonth.csv</i> &amp; <i>data/Traffic.csv</i> (Kaggle dataset)<br/>"
            "- <i>models/knn_traffic_model.pkl</i> (Trained KNN model &amp; StandardScaler)<br/>"
            "- <i>outputs/knn_evaluation.png</i> &amp; <i>outputs/route_comparison.png</i> (Charts)<br/>"
            "- <i>SmartPulse_AI_Project_Report.pdf</i> (This complete lab report)",
            body,
        )
    )

    doc.build(story)
    print(f"[PDF] Generated complete project report at: {pdf_path}")
    return pdf_path
