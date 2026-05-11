#!/usr/bin/env python3
"""
Interactive live demo for Greedy vs Christofides on Euclidean TSP instances.

Designed for classroom or presentation use:
- One-click random/clustered instance generation
- Stage-by-stage Christofides breakdown
- Step-by-step greedy construction
- Live quality comparison against exact optimum (for n <= 12) or MST lower bound
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live TSP demo for Greedy and Christofides.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build one demo instance with a non-interactive backend and exit.",
    )
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="Optional path to save a snapshot image while running smoke-test.",
    )
    return parser.parse_args()


ARGS = parse_args()
if ARGS.smoke_test:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from christofides import christofides_tsp
from exact import held_karp_tsp
from generate import distance_matrix, generate_points_clustered, generate_points_random
from greedy import greedy_tsp
from metrics import mst_lower_bound

STAGE_ORDER = ["overview", "greedy", "mst", "matching", "euler", "final"]
STAGE_LABELS = {
    "overview": "Overview",
    "greedy": "Greedy Build",
    "mst": "Christofides: MST",
    "matching": "Christofides: Odd Nodes + Matching",
    "euler": "Christofides: Euler Tour",
    "final": "Final Comparison",
}
STAGE_HELP = {
    "overview": "Start here: both panels show the same Euclidean TSP instance.",
    "greedy": "Greedy picks the nearest unvisited city at each step. Use the progress slider to reveal each choice.",
    "mst": "Christofides begins with a minimum spanning tree, which is also a lower bound on the tour cost.",
    "matching": "Odd-degree MST vertices are paired with a minimum-weight perfect matching to make an Eulerian multigraph.",
    "euler": "An Eulerian walk uses every edge exactly once. Shortcutting repeated vertices preserves metric feasibility.",
    "final": "Final tours are shown side by side. The report's key message is the tradeoff: Greedy is faster, Christofides is usually better.",
}


@dataclass
class ChristofidesBreakdown:
    mst_edges: list[tuple[int, int]]
    odd_vertices: list[int]
    matching_edges: list[tuple[int, int]]
    euler_walk: list[int]
    final_tour: list[int]
    final_length: float


@dataclass
class DemoState:
    points: np.ndarray
    dist_matrix: np.ndarray
    instance_type: str
    seed: int
    greedy_tour: list[int]
    greedy_length: float
    christofides: ChristofidesBreakdown
    mst_cost: float
    optimum_cost: float | None
    greedy_ratio: float
    christofides_ratio: float
    ratio_reference: str


def tour_length(dist_matrix: np.ndarray, tour: Iterable[int]) -> float:
    ordered = list(tour)
    return float(sum(dist_matrix[ordered[i], ordered[i + 1]] for i in range(len(ordered) - 1)))


def christofides_breakdown(dist_matrix: np.ndarray) -> ChristofidesBreakdown:
    n = len(dist_matrix)
    if n == 0:
        return ChristofidesBreakdown([], [], [], [], [], 0.0)
    if n == 1:
        return ChristofidesBreakdown([], [], [], [0, 0], [0, 0], 0.0)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            graph.add_edge(i, j, weight=float(dist_matrix[i, j]))

    mst = nx.minimum_spanning_tree(graph, weight="weight")
    odd_vertices = [node for node in mst.nodes if mst.degree(node) % 2 == 1]

    odd_graph = nx.Graph()
    odd_graph.add_nodes_from(odd_vertices)
    for index, u in enumerate(odd_vertices):
        for v in odd_vertices[index + 1 :]:
            odd_graph.add_edge(u, v, weight=float(dist_matrix[u, v]))

    matching = nx.min_weight_matching(odd_graph, weight="weight")
    matching_edges = [tuple(edge) for edge in matching]

    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(range(n))
    for u, v in mst.edges():
        multigraph.add_edge(u, v, weight=float(dist_matrix[u, v]))
    for u, v in matching_edges:
        multigraph.add_edge(u, v, weight=float(dist_matrix[u, v]))

    euler_edges = list(nx.eulerian_circuit(multigraph, source=0))
    euler_walk = [euler_edges[0][0]] + [v for _, v in euler_edges]

    final_tour = []
    visited = set()
    for node in euler_walk:
        if node not in visited:
            final_tour.append(node)
            visited.add(node)
    final_tour.append(final_tour[0])

    return ChristofidesBreakdown(
        mst_edges=[tuple(edge) for edge in mst.edges()],
        odd_vertices=odd_vertices,
        matching_edges=matching_edges,
        euler_walk=euler_walk,
        final_tour=final_tour,
        final_length=tour_length(dist_matrix, final_tour),
    )


def build_demo_state(n: int, instance_type: str, seed: int) -> DemoState:
    if instance_type == "random":
        points = generate_points_random(n, seed)
    else:
        points = generate_points_clustered(n, seed=seed)

    dist_matrix = distance_matrix(points)
    greedy_tour, greedy_length = greedy_tsp(dist_matrix)
    christo = christofides_breakdown(dist_matrix)
    mst_cost = float(mst_lower_bound(dist_matrix))

    optimum_cost = None
    ratio_reference = "MST lower bound"
    reference_cost = mst_cost
    if n <= 12:
        _, optimum_cost = held_karp_tsp(dist_matrix)
        ratio_reference = "exact optimum"
        reference_cost = optimum_cost

    return DemoState(
        points=points,
        dist_matrix=dist_matrix,
        instance_type=instance_type,
        seed=seed,
        greedy_tour=greedy_tour,
        greedy_length=float(greedy_length),
        christofides=christo,
        mst_cost=mst_cost,
        optimum_cost=optimum_cost,
        greedy_ratio=float(greedy_length / reference_cost) if reference_cost else 0.0,
        christofides_ratio=float(christo.final_length / reference_cost) if reference_cost else 0.0,
        ratio_reference=ratio_reference,
    )


class LiveTspDemo:
    def __init__(self) -> None:
        self.stage_index = 0
        self.seed = 42
        self.instance_type = "random"
        self.city_count = 10
        self.state = build_demo_state(self.city_count, self.instance_type, self.seed)

        self.fig = plt.figure(figsize=(15, 8.8))
        self.left_ax = self.fig.add_axes([0.05, 0.24, 0.4, 0.64])
        self.right_ax = self.fig.add_axes([0.54, 0.24, 0.4, 0.64])

        self.prev_button = Button(self.fig.add_axes([0.05, 0.08, 0.1, 0.05]), "Prev Stage")
        self.next_button = Button(self.fig.add_axes([0.16, 0.08, 0.1, 0.05]), "Next Stage")
        self.random_button = Button(self.fig.add_axes([0.31, 0.08, 0.12, 0.05]), "New Random")
        self.cluster_button = Button(self.fig.add_axes([0.44, 0.08, 0.14, 0.05]), "New Clustered")
        self.cities_slider = Slider(
            ax=self.fig.add_axes([0.64, 0.095, 0.28, 0.03]),
            label="Cities",
            valmin=6,
            valmax=18,
            valinit=self.city_count,
            valstep=1,
        )
        self.greedy_slider = Slider(
            ax=self.fig.add_axes([0.64, 0.045, 0.28, 0.03]),
            label="Greedy Progress",
            valmin=1,
            valmax=max(1, len(self.state.greedy_tour) - 1),
            valinit=max(1, len(self.state.greedy_tour) - 1),
            valstep=1,
        )

        self.prev_button.on_clicked(self.prev_stage)
        self.next_button.on_clicked(self.next_stage)
        self.random_button.on_clicked(self.new_random)
        self.cluster_button.on_clicked(self.new_clustered)
        self.cities_slider.on_changed(self.change_city_count)
        self.greedy_slider.on_changed(self.redraw)

        self.redraw()

    def next_stage(self, _event=None) -> None:
        self.stage_index = min(self.stage_index + 1, len(STAGE_ORDER) - 1)
        self.redraw()

    def prev_stage(self, _event=None) -> None:
        self.stage_index = max(self.stage_index - 1, 0)
        self.redraw()

    def new_random(self, _event=None) -> None:
        self.seed += 1
        self.instance_type = "random"
        self.refresh_state()

    def new_clustered(self, _event=None) -> None:
        self.seed += 1
        self.instance_type = "clustered"
        self.refresh_state()

    def change_city_count(self, value: float) -> None:
        new_count = int(value)
        if new_count != self.city_count:
            self.city_count = new_count
            self.refresh_state()

    def refresh_state(self) -> None:
        self.state = build_demo_state(self.city_count, self.instance_type, self.seed)
        self.stage_index = 0

        self.greedy_slider.eventson = False
        self.greedy_slider.valmax = max(1, len(self.state.greedy_tour) - 1)
        self.greedy_slider.ax.set_xlim(self.greedy_slider.valmin, self.greedy_slider.valmax)
        self.greedy_slider.set_val(self.greedy_slider.valmax)
        self.greedy_slider.eventson = True

        self.redraw()

    def redraw(self, _value=None) -> None:
        stage = STAGE_ORDER[self.stage_index]
        self.fig.suptitle(
            f"TSP Live Demo: Greedy vs Christofides | {STAGE_LABELS[stage]}",
            fontsize=18,
            fontweight="bold",
        )

        self.draw_panel(self.left_ax, "greedy", stage)
        self.draw_panel(self.right_ax, "christofides", stage)
        self.draw_footer(stage)
        self.fig.canvas.draw_idle()

    def draw_panel(self, ax: plt.Axes, algorithm: str, stage: str) -> None:
        ax.clear()
        points = self.state.points
        ax.set_facecolor("#f8fafc")
        ax.grid(alpha=0.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.scatter(points[:, 0], points[:, 1], s=65, c="#111827", zorder=3)
        ax.scatter(points[0, 0], points[0, 1], s=120, c="#ef4444", marker="*", zorder=4)

        for index, (x_coord, y_coord) in enumerate(points):
            ax.text(x_coord + 0.012, y_coord + 0.012, str(index), fontsize=9, color="#1f2937")

        if algorithm == "greedy":
            ax.set_title("Greedy (Nearest Neighbor)", fontsize=14, fontweight="bold")
            self.draw_greedy(ax, stage)
        else:
            ax.set_title("Christofides", fontsize=14, fontweight="bold")
            self.draw_christofides(ax, stage)

    def draw_edges(
        self,
        ax: plt.Axes,
        edges: Iterable[tuple[int, int]],
        *,
        color: str,
        linewidth: float = 2.5,
        linestyle: str = "-",
        alpha: float = 0.95,
    ) -> None:
        for u, v in edges:
            x_values = [self.state.points[u, 0], self.state.points[v, 0]]
            y_values = [self.state.points[u, 1], self.state.points[v, 1]]
            ax.plot(x_values, y_values, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=2)

    def draw_path(self, ax: plt.Axes, tour: list[int], color: str) -> None:
        for i in range(len(tour) - 1):
            self.draw_edges(ax, [(tour[i], tour[i + 1])], color=color, linewidth=2.8)

    def draw_greedy(self, ax: plt.Axes, stage: str) -> None:
        if stage == "overview":
            ax.text(
                0.03,
                0.96,
                "Local rule:\nPick the nearest unvisited city.",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round", "facecolor": "#fff7ed", "edgecolor": "#fdba74"},
            )
            return

        if stage == "greedy":
            edge_count = int(self.greedy_slider.val)
            edge_count = min(edge_count, len(self.state.greedy_tour) - 1)
            partial_edges = [
                (self.state.greedy_tour[i], self.state.greedy_tour[i + 1])
                for i in range(edge_count)
            ]
            self.draw_edges(ax, partial_edges, color="#f97316")
            if partial_edges:
                current_city = self.state.greedy_tour[min(edge_count, len(self.state.greedy_tour) - 1)]
                ax.text(
                    0.03,
                    0.96,
                    f"Current city: {current_city}\nEdges revealed: {edge_count}/{len(self.state.greedy_tour) - 1}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=11,
                    bbox={"boxstyle": "round", "facecolor": "#fff7ed", "edgecolor": "#fb923c"},
                )
            return

        self.draw_path(ax, self.state.greedy_tour, "#f97316")
        ax.text(
            0.03,
            0.96,
            f"Final greedy tour length: {self.state.greedy_length:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "#fff7ed", "edgecolor": "#fb923c"},
        )

    def draw_christofides(self, ax: plt.Axes, stage: str) -> None:
        breakdown = self.state.christofides
        if stage == "overview":
            ax.text(
                0.03,
                0.96,
                "Pipeline:\nMST -> odd nodes -> matching -> Euler walk -> shortcutting",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
            )
            return

        if stage == "greedy":
            ax.text(
                0.03,
                0.96,
                "Christofides panel will activate in the next stage.",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
            )
            return

        self.draw_edges(ax, breakdown.mst_edges, color="#0f766e")

        if stage == "mst":
            ax.text(
                0.03,
                0.96,
                f"MST cost = {self.state.mst_cost:.3f}\nThis is a lower bound on any valid tour.",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round", "facecolor": "#ecfdf5", "edgecolor": "#6ee7b7"},
            )
            return

        odd = breakdown.odd_vertices
        ax.scatter(
            self.state.points[odd, 0],
            self.state.points[odd, 1],
            s=160,
            facecolors="none",
            edgecolors="#ef4444",
            linewidths=2,
            zorder=4,
        )

        if stage == "matching":
            self.draw_edges(ax, breakdown.matching_edges, color="#2563eb", linestyle="--")
            ax.text(
                0.03,
                0.96,
                f"Odd MST vertices: {len(odd)}\nMatching edges repair parity so an Euler tour exists.",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
            )
            return

        if stage == "euler":
            euler_edges = [
                (breakdown.euler_walk[i], breakdown.euler_walk[i + 1])
                for i in range(len(breakdown.euler_walk) - 1)
            ]
            self.draw_edges(ax, euler_edges, color="#2563eb", alpha=0.55)
            ax.text(
                0.03,
                0.96,
                f"Euler walk visits repeated vertices.\nLength before shortcutting: {tour_length(self.state.dist_matrix, breakdown.euler_walk):.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
            )
            return

        self.draw_path(ax, breakdown.final_tour, "#2563eb")
        ax.text(
            0.03,
            0.96,
            f"Final Christofides tour length: {breakdown.final_length:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
        )

    def draw_footer(self, stage: str) -> None:
        for artist in list(self.fig.texts):
            if getattr(artist, "_demo_footer", False):
                artist.remove()

        best_name = "Christofides" if self.state.christofides.final_length < self.state.greedy_length else "Greedy"
        info_lines = [
            f"Instance: {self.instance_type.title()} | cities = {self.city_count} | seed = {self.seed}",
            f"Greedy length = {self.state.greedy_length:.3f} | ratio vs {self.state.ratio_reference} = {self.state.greedy_ratio:.3f}",
            f"Christofides length = {self.state.christofides.final_length:.3f} | ratio vs {self.state.ratio_reference} = {self.state.christofides_ratio:.3f}",
            f"MST lower bound = {self.state.mst_cost:.3f}"
            + (f" | exact optimum = {self.state.optimum_cost:.3f}" if self.state.optimum_cost is not None else ""),
            f"Complexity story: Greedy ~ O(n^2), Christofides ~ O(n^3). Quality winner on this instance: {best_name}.",
            f"Presenter cue: {STAGE_HELP[stage]}",
        ]

        y = 0.195
        for line in info_lines:
            text = self.fig.text(0.05, y, line, fontsize=11, ha="left", va="top", color="#111827")
            text._demo_footer = True
            y -= 0.028

    def show(self) -> None:
        plt.show()


def run_smoke_test(snapshot_path: Path | None) -> int:
    demo = LiveTspDemo()
    demo.stage_index = len(STAGE_ORDER) - 1
    demo.redraw()

    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        demo.fig.savefig(snapshot_path, dpi=150, bbox_inches="tight")

    print(
        "SMOKE_OK",
        f"cities={demo.city_count}",
        f"instance={demo.instance_type}",
        f"greedy={demo.state.greedy_length:.3f}",
        f"christofides={demo.state.christofides.final_length:.3f}",
        f"reference={demo.state.ratio_reference}",
    )
    plt.close(demo.fig)
    return 0


def main() -> int:
    if ARGS.smoke_test:
        return run_smoke_test(ARGS.save_snapshot)

    demo = LiveTspDemo()
    demo.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
