"""Main entry point for SmartPulse AI: Kaggle Traffic KNN + A* Emergency Router."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.city_router import plot_route_comparison, run_dispatch_scenario
from src.download_data import download_kaggle_dataset
from src.knn_model import train_and_evaluate_knn
from src.pdf_report import generate_pdf_report

VALID_DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for running or customizing the lab pipeline."""
    parser = argparse.ArgumentParser(
        description="SmartPulse AI: Kaggle KNN Traffic Prediction + A* vs BFS Ambulance Routing"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading the dataset from kaggle.com even if cached.",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help="Optional custom dispatch hour (0-23) to simulate an extra scenario.",
    )
    parser.add_argument(
        "--day",
        type=str,
        default="Friday",
        choices=VALID_DAYS,
        help="Day of week for the optional custom scenario (default: Friday).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip generating the PDF report.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute end-to-end training, search benchmarking, plotting, and PDF reporting."""
    args = parse_args()
    root = Path(__file__).resolve().parent

    data_dir = root / "data"
    model_path = root / "models" / "knn_traffic_model.pkl"
    knn_plot_path = root / "outputs" / "knn_evaluation.png"
    route_plot_path = root / "outputs" / "route_comparison.png"
    pdf_path = root / "SmartPulse_AI_Project_Report.pdf"

    print("=" * 68)
    print("  SmartPulse AI Lab: KNN Traffic Prediction + A* & BFS City Router")
    print("=" * 68)

    # 1. Download dataset from Kaggle
    csv_path = download_kaggle_dataset(data_dir, force=args.force_download)

    # 2. Train and evaluate KNN model
    bundle = train_and_evaluate_knn(csv_path, model_path, knn_plot_path)
    metrics = bundle["metrics"]

    # 3. Run A* vs BFS dispatch scenarios using KNN-predicted road costs
    scenarios = [
        run_dispatch_scenario(
            bundle, name="Morning Rush Hour (Mon 08:00)", hour=8, day_name="Monday"
        ),
        run_dispatch_scenario(
            bundle, name="Late Night Off-Peak (Sun 02:00)", hour=2, day_name="Sunday"
        ),
    ]

    if args.hour is not None:
        if not 0 <= args.hour <= 23:
            raise ValueError("--hour must be between 0 and 23")
        scenarios.append(
            run_dispatch_scenario(
                bundle,
                name=f"Custom Dispatch ({args.day[:3]} {args.hour:02d}:00)",
                hour=args.hour,
                day_name=args.day,
            )
        )

    plot_route_comparison(scenarios, route_plot_path)

    print("\n--- Dispatch Routing Summary (Depot -> Patient -> Hospital) ---")
    for scen in scenarios:
        a = scen["astar"]
        b = scen["bfs"]
        print(
            f"  {scen['name']}:\n"
            f"    A* + KNN : {a.total_cost:5.1f} min | {a.total_hops:2d} hops | {a.nodes_explored:3d} nodes explored\n"
            f"    BFS      : {b.total_cost:5.1f} min | {b.total_hops:2d} hops | {b.nodes_explored:3d} nodes explored\n"
            f"    Savings  : {scen['time_saved_min']:.1f} min faster ({scen['time_saved_pct']:.1f}% reduction)"
        )

    # 4. Generate complete PDF report
    if not args.no_pdf:
        generate_pdf_report(
            pdf_path=pdf_path,
            metrics=metrics,
            scenarios=scenarios,
            knn_plot_path=knn_plot_path,
            route_plot_path=route_plot_path,
        )

    print("\nAll outputs generated successfully in:", root)


if __name__ == "__main__":
    main()
