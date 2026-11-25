"""City grid simulation with KNN traffic prediction, A* Search, and BFS comparison."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.knn_model import engineer_features

GRID_SIZE = 16
DEPOT = (1, 1)
PATIENT = (8, 11)
HOSPITAL = (14, 14)

CONGESTION_COSTS = {
    "low": 1.0,
    "normal": 2.0,
    "high": 4.0,
    "heavy": 7.5,
}
CONGESTION_CODES = {
    "blocked": 0,
    "low": 1,
    "normal": 2,
    "high": 3,
    "heavy": 4,
}
MIN_STEP_COST = min(CONGESTION_COSTS.values())

# City building blocks (impassable obstacles) placed to leave central + ring corridors
OBSTACLES: set[tuple[int, int]] = {
    (2, 4), (2, 5), (3, 4), (3, 5),
    (2, 10), (2, 11), (3, 10), (3, 11),
    (5, 2), (6, 2), (5, 3), (6, 3),
    (5, 7), (5, 8), (6, 7), (6, 8),
    (9, 4), (9, 5), (10, 4), (10, 5),
    (9, 9), (10, 9), (10, 10),
    (12, 2), (12, 3), (13, 2), (13, 3),
    (12, 7), (12, 8), (13, 7), (13, 8),
    (11, 13), (12, 13),
}


@dataclass
class SearchResult:
    """Stores pathfinding outcome for a single algorithm across a multi-leg mission."""

    algorithm: str
    path_leg1: list[tuple[int, int]]
    path_leg2: list[tuple[int, int]]
    total_cost: float
    total_hops: int
    nodes_explored: int

    @property
    def full_path(self) -> list[tuple[int, int]]:
        """Combined path from Depot -> Patient -> Hospital."""
        if not self.path_leg1:
            return self.path_leg2
        if not self.path_leg2:
            return self.path_leg1
        return self.path_leg1 + self.path_leg2[1:]


def _neighbors(pos: tuple[int, int], grid_size: int, obstacles: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return valid 4-directional neighbor coordinates."""
    r, c = pos
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (nr, nc)
        for nr, nc in candidates
        if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in obstacles
    ]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Admissible Manhattan distance heuristic scaled by minimum edge cost."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1])) * MIN_STEP_COST


def astar_search(
    cost_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: set[tuple[int, int]] = OBSTACLES,
) -> tuple[list[tuple[int, int]], float, int]:
    """Find minimum-travel-time path from start to goal using A* Search.

    Uses f(n) = g(n) + h(n) where g(n) is cumulative KNN-predicted travel delay
    and h(n) is the admissible Manhattan distance * MIN_STEP_COST.
    """
    grid_size = cost_grid.shape[0]
    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (_manhattan(start, goal), counter, start))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    nodes_explored = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_explored += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, float(g_score[goal]), nodes_explored

        for nxt in _neighbors(current, grid_size, obstacles):
            if nxt in closed:
                continue
            tentative_g = g_score[current] + float(cost_grid[nxt])
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + _manhattan(nxt, goal)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, nxt))

    raise RuntimeError(f"A* failed to find a valid path from {start} to {goal}")


def bfs_search(
    cost_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: set[tuple[int, int]] = OBSTACLES,
) -> tuple[list[tuple[int, int]], float, int]:
    """Find fewest-hop path using Breadth-First Search and compute its delay cost."""
    grid_size = cost_grid.shape[0]
    queue: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    visited: set[tuple[int, int]] = {start}
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            total_cost = float(sum(cost_grid[cell] for cell in path[1:]))
            return path, total_cost, nodes_explored

        for nxt in _neighbors(current, grid_size, obstacles):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                queue.append(nxt)

    raise RuntimeError(f"BFS failed to find a valid path from {start} to {goal}")


def build_knn_traffic_grid(
    bundle: dict[str, Any],
    hour: int,
    day_name: str,
    grid_size: int = GRID_SIZE,
    obstacles: set[tuple[int, int]] = OBSTACLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Populate a city grid with real Kaggle test records and predict risk via KNN.

    Central downtown corridors sample from higher-density records in the target
    time window, while outer ring roads sample from lighter records in the same
    time window. Every drivable cell is classified by the trained KNN model.
    """
    test_df: pd.DataFrame = bundle["test_records"].copy()
    parsed_time = pd.to_datetime(test_df["Time"], format="%I:%M:%S %p")
    test_df["_hour"] = parsed_time.dt.hour
    test_df["_is_weekend"] = test_df["Day of the week"].isin(["Saturday", "Sunday"])

    target_weekend = day_name in ("Saturday", "Sunday")
    hour_diff = np.minimum(
        np.abs(test_df["_hour"] - hour),
        24 - np.abs(test_df["_hour"] - hour),
    )

    window_df = test_df[(hour_diff <= 2) & (test_df["_is_weekend"] == target_weekend)]
    if len(window_df) < 20:
        window_df = test_df[hour_diff <= 3]
    if len(window_df) < 20:
        window_df = test_df

    window_sorted = window_df.sort_values("Total", ascending=True).reset_index(drop=True)
    n_rows = len(window_sorted)
    downtown_pool = window_sorted.iloc[max(0, int(n_rows * 0.45)):].reset_index(drop=True)
    ring_pool = window_sorted.iloc[:max(1, int(n_rows * 0.55))].reset_index(drop=True)

    drivable_cells: list[tuple[int, int]] = []
    sampled_rows: list[pd.Series] = []

    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) in obstacles:
                continue
            drivable_cells.append((r, c))
            # Central commercial zone along the direct Depot -> Patient -> Hospital diagonal
            is_central = (3 <= r <= 12) and (3 <= c <= 12)
            pool = downtown_pool if is_central else ring_pool
            idx = (r * 37 + c * 19 + hour * 7) % len(pool)
            sampled_rows.append(pool.iloc[idx])

    cell_df = pd.DataFrame(sampled_rows).reset_index(drop=True)
    cell_features = engineer_features(cell_df)
    cell_scaled = bundle["scaler"].transform(cell_features)
    predictions = bundle["model"].predict(cell_scaled)

    cost_grid = np.full((grid_size, grid_size), np.inf, dtype=float)
    label_grid = np.full((grid_size, grid_size), "blocked", dtype=object)

    for (r, c), pred in zip(drivable_cells, predictions):
        label_grid[r, c] = str(pred)
        cost_grid[r, c] = CONGESTION_COSTS[str(pred)]

    return cost_grid, label_grid


def run_dispatch_scenario(
    bundle: dict[str, Any],
    name: str,
    hour: int,
    day_name: str,
) -> dict[str, Any]:
    """Execute A* and BFS dispatch routing on a KNN-predicted city traffic grid."""
    cost_grid, label_grid = build_knn_traffic_grid(bundle, hour=hour, day_name=day_name)

    a1_path, a1_cost, a1_exp = astar_search(cost_grid, DEPOT, PATIENT)
    a2_path, a2_cost, a2_exp = astar_search(cost_grid, PATIENT, HOSPITAL)
    astar_res = SearchResult(
        algorithm="A* Search",
        path_leg1=a1_path,
        path_leg2=a2_path,
        total_cost=a1_cost + a2_cost,
        total_hops=(len(a1_path) - 1) + (len(a2_path) - 1),
        nodes_explored=a1_exp + a2_exp,
    )

    b1_path, b1_cost, b1_exp = bfs_search(cost_grid, DEPOT, PATIENT)
    b2_path, b2_cost, b2_exp = bfs_search(cost_grid, PATIENT, HOSPITAL)
    bfs_res = SearchResult(
        algorithm="BFS Baseline",
        path_leg1=b1_path,
        path_leg2=b2_path,
        total_cost=b1_cost + b2_cost,
        total_hops=(len(b1_path) - 1) + (len(b2_path) - 1),
        nodes_explored=b1_exp + b2_exp,
    )

    time_saved = bfs_res.total_cost - astar_res.total_cost
    pct_saved = (time_saved / bfs_res.total_cost * 100.0) if bfs_res.total_cost > 0 else 0.0

    # Count cell congestion distribution for reporting
    unique, counts = np.unique(label_grid, return_counts=True)
    grid_dist = {str(k): int(v) for k, v in zip(unique, counts) if k != "blocked"}

    return {
        "name": name,
        "hour": hour,
        "day_name": day_name,
        "cost_grid": cost_grid,
        "label_grid": label_grid,
        "grid_distribution": grid_dist,
        "astar": astar_res,
        "bfs": bfs_res,
        "time_saved_min": time_saved,
        "time_saved_pct": pct_saved,
    }


def _draw_city_panel(ax: plt.Axes, scenario: dict[str, Any]) -> None:
    """Render one city traffic heatmap with overlaid A* and BFS routes."""
    label_grid = scenario["label_grid"]
    code_grid = np.vectorize(lambda lbl: CONGESTION_CODES[lbl])(label_grid)

    cmap = ListedColormap(["#2c3e50", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
    ax.imshow(code_grid, cmap=cmap, vmin=0, vmax=4, origin="upper")

    astar_path = scenario["astar"].full_path
    bfs_path = scenario["bfs"].full_path

    bfs_r, bfs_c = zip(*bfs_path)
    astar_r, astar_c = zip(*astar_path)

    ax.plot(bfs_c, bfs_r, linestyle="--", linewidth=2.6, color="#8e44ad", label="BFS (Shortest Hops)")
    ax.plot(astar_c, astar_r, linestyle="-", linewidth=2.8, color="#00d2ff", label="A* + KNN (Fastest)")

    # Key landmarks
    for (r, c), txt, color in [
        (DEPOT, "D", "#ffffff"),
        (PATIENT, "P", "#ffffff"),
        (HOSPITAL, "H", "#ffffff"),
    ]:
        ax.scatter(c, r, s=160, color="#111111", edgecolors=color, linewidths=1.8, zorder=6)
        ax.text(c, r, txt, color=color, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=7)

    a_cost = scenario["astar"].total_cost
    b_cost = scenario["bfs"].total_cost
    ax.set_title(
        f"{scenario['name']}\nA*: {a_cost:.1f} min ({scenario['astar'].total_hops} hops) | "
        f"BFS: {b_cost:.1f} min ({scenario['bfs'].total_hops} hops)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.4, alpha=0.4)


def plot_route_comparison(scenarios: list[dict[str, Any]], output_path: Path) -> Path:
    """Save a multi-panel figure comparing KNN traffic heatmaps and A* vs BFS metrics."""
    n_scen = len(scenarios)
    fig, axes = plt.subplots(1, n_scen + 1, figsize=(6 * n_scen + 5.2, 5.6), dpi=150)

    for idx, scen in enumerate(scenarios):
        _draw_city_panel(axes[idx], scen)

    # Shared map legend placed cleanly below the grid panels
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Low (1.0m)"),
        Patch(facecolor="#f1c40f", label="Normal (2.0m)"),
        Patch(facecolor="#e67e22", label="High (4.0m)"),
        Patch(facecolor="#e74c3c", label="Heavy (7.5m)"),
        Patch(facecolor="#2c3e50", label="Building"),
    ]
    route_handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=legend_elements + route_handles,
        loc="lower center",
        bbox_to_anchor=(0.36, -0.04),
        ncol=4,
        fontsize=8.5,
        framealpha=0.95,
    )

    # Bar comparison panel
    ax_bar = axes[-1]
    labels = [s["name"].split("(")[0].strip() for s in scenarios]
    x = np.arange(len(labels))
    width = 0.34
    astar_costs = [s["astar"].total_cost for s in scenarios]
    bfs_costs = [s["bfs"].total_cost for s in scenarios]

    bars1 = ax_bar.bar(x - width / 2, astar_costs, width, label="A* + KNN", color="#0099cc")
    bars2 = ax_bar.bar(x + width / 2, bfs_costs, width, label="BFS Baseline", color="#8e44ad")

    ax_bar.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8.5)
    ax_bar.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8.5)
    ax_bar.set_title("Total Travel Delay Comparison", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Estimated Travel Time (minutes)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylim(0, max(bfs_costs) * 1.2)
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
    ax_bar.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
