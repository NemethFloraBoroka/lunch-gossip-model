"""
Lunch Gossip Model
==================

A simple agent-based temporal network model for gossip spreading in a workplace.

Core idea
---------
There is a stable workplace friendship/acquaintance graph. This graph shapes the
lunch seating process: direct acquaintances, and to a weaker extent two-step
acquaintances, are more likely to sit together. Gossip itself is transmitted
during lunch. Once an active spreader talks at a table, any susceptible tablemate
may hear the gossip. Transmission probability depends on:

- how close the two people sit at the same circular table,
- a global gossip transmission probability.

Thus, the acquaintance graph affects contact formation, while the lunch table
creates the actual exposure situation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Set, Literal

import math
import random
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300


GraphType = Literal["erdos_renyi", "barabasi_albert", "watts_strogatz", "sbm"]
StartMode = Literal["random", "highest_degree", "lowest_degree", "highest_betweenness"]
RumorModel = Literal["SI", "SIR"]


@dataclass
class LunchGossipParams:
    """Parameters controlling the model.

    Parameters
    ----------
    n_people:
        Number of employees.
    avg_degree:
        Approximate average degree in the workplace acquaintance graph.
    table_size:
        Maximum number of people at a lunch table.
    n_tables:
        Preferred fixed number of lunch tables. If None, the minimum necessary
        number of tables is used. If the provided value is too small for the
        number of lunch attendees, the model uses the required minimum and emits
        a warning.
    beta:
        Baseline probability of transmitting the gossip, before modifiers.
    seating_factor:
        Weight of two-step acquaintances in the seating process. Direct
        acquaintances contribute 1.0 to seating preference, two-step
        acquaintances contribute seating_factor, and unrelated people contribute
        0.0. This parameter affects where people sit, not who can hear the gossip.
    seating_alpha:
        Controls how strongly transmission probability decreases with circular
        seat distance. Larger values mean stronger decay.
    friend_seating_bias:
        Controls how strongly the seating algorithm prefers putting socially
        close people together. 0 means fully random seating; larger values create
        more socially clustered tables.
    lunch_attendance_prob:
        Probability that an employee appears at lunch on a given day.
    rumor_model:
        "SI" means informed people keep spreading forever.
        "SIR" means informed spreaders may stop spreading with probability
        stifling_prob after each day.
    stifling_prob:
        Daily probability that an active spreader becomes inactive in SIR mode.
    graph_type:
        Type of acquaintance graph to generate.
    n_groups:
        Number of workplace groups/departments. In SBM mode, these define blocks.
        These groups are only used to create modularity in the acquaintance graph;
        all employees still share the same cafeteria.
    p_intra:
        Within-group connection probability for SBM. If None, estimated from
        avg_degree.
    p_inter:
        Between-group connection probability for SBM. If None, set much lower
        than p_intra.
    seed:
        Random seed for reproducibility.
    """

    n_people: int = 200
    avg_degree: int = 8
    table_size: int = 6
    n_tables: Optional[int] = None
    beta: float = 0.25
    seating_factor: float = 0.25
    seating_alpha: float = 0.7
    friend_seating_bias: float = 2.0
    lunch_attendance_prob: float = 1.0
    rumor_model: RumorModel = "SI"
    stifling_prob: float = 0.05
    graph_type: GraphType = "sbm"
    n_groups: int = 1
    p_intra: Optional[float] = None
    p_inter: Optional[float] = None
    seed: Optional[int] = None


class LunchGossipModel:
    """Agent-based model of workplace gossip spreading through lunch contacts."""

    def __init__(self, params: LunchGossipParams):
        self.params = params
        self.rng = random.Random(params.seed)
        self.np_rng = np.random.default_rng(params.seed)

        self.friendship_graph: nx.Graph = self._generate_friendship_graph()
        self.groups: Dict[int, int] = nx.get_node_attributes(
            self.friendship_graph,
            "group",
        )

        if not self.groups:
            self.groups = {i: 0 for i in self.friendship_graph.nodes}
            nx.set_node_attributes(self.friendship_graph, self.groups, "group")

        self.day: int = 0
        self.state: Dict[int, str] = {
            i: "S" for i in self.friendship_graph.nodes
        }
        self.infection_day: Dict[int, Optional[int]] = {
            i: None for i in self.friendship_graph.nodes
        }
        self.current_tables: List[List[int]] = []
        self.event_log: List[Dict] = []
        self.history: List[Dict] = []

    # ---------------------------------------------------------------------
    # Graph generation
    # ---------------------------------------------------------------------

    def _generate_friendship_graph(self) -> nx.Graph:
        p = self.params
        n = p.n_people

        if p.graph_type == "erdos_renyi":
            prob = min(1.0, p.avg_degree / max(1, n - 1))
            G = nx.erdos_renyi_graph(n=n, p=prob, seed=p.seed)
            nx.set_node_attributes(G, {i: 0 for i in G.nodes}, "group")

        elif p.graph_type == "barabasi_albert":
            m = max(1, int(round(p.avg_degree / 2)))
            m = min(m, n - 1)
            G = nx.barabasi_albert_graph(n=n, m=m, seed=p.seed)
            nx.set_node_attributes(G, {i: 0 for i in G.nodes}, "group")

        elif p.graph_type == "watts_strogatz":
            k = max(2, int(round(p.avg_degree)))
            if k % 2 == 1:
                k += 1
            k = min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
            G = nx.watts_strogatz_graph(n=n, k=k, p=0.1, seed=p.seed)
            nx.set_node_attributes(G, {i: 0 for i in G.nodes}, "group")

        elif p.graph_type == "sbm":
            n_groups = max(1, p.n_groups)
            sizes = self._balanced_block_sizes(n, n_groups)

            if p.p_intra is None:
                average_block_size = n / n_groups
                p_intra = min(
                    1.0,
                    p.avg_degree / max(1, average_block_size - 1),
                )
            else:
                p_intra = p.p_intra

            if p.p_inter is None:
                p_inter = min(p_intra * 0.08, p_intra)
            else:
                p_inter = p.p_inter

            probs = [
                [p_intra if i == j else p_inter for j in range(n_groups)]
                for i in range(n_groups)
            ]

            G = nx.stochastic_block_model(sizes, probs, seed=p.seed)

            group_attr = {}
            start = 0
            for group_id, size in enumerate(sizes):
                for node in range(start, start + size):
                    group_attr[node] = group_id
                start += size

            nx.set_node_attributes(G, group_attr, "group")

        else:
            raise ValueError(f"Unknown graph_type: {p.graph_type}")

        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    @staticmethod
    def _balanced_block_sizes(n: int, n_blocks: int) -> List[int]:
        base = n // n_blocks
        remainder = n % n_blocks
        return [
            base + (1 if i < remainder else 0)
            for i in range(n_blocks)
        ]

    # ---------------------------------------------------------------------
    # Initialization and simulation
    # ---------------------------------------------------------------------

    def reset(
        self,
        start_node: Optional[int] = None,
        start_mode: StartMode = "random",
    ) -> int:
        """Reset the simulation and initialize one person as informed."""
        self.day = 0
        self.state = {i: "S" for i in self.friendship_graph.nodes}
        self.infection_day = {
            i: None for i in self.friendship_graph.nodes
        }
        self.current_tables = []
        self.event_log = []
        self.history = []

        if start_node is None:
            start_node = self.choose_start_node(start_mode)

        self.state[start_node] = "I"
        self.infection_day[start_node] = 0
        self._record_history(newly_informed=[start_node], transmissions=[])

        return start_node

    def choose_start_node(self, mode: StartMode = "random") -> int:
        G = self.friendship_graph

        if mode == "random":
            return self.rng.choice(list(G.nodes))

        if mode == "highest_degree":
            return max(G.nodes, key=lambda x: G.degree[x])

        if mode == "lowest_degree":
            return min(G.nodes, key=lambda x: G.degree[x])

        if mode == "highest_betweenness":
            bc = nx.betweenness_centrality(G)
            return max(G.nodes, key=lambda x: bc[x])

        raise ValueError(f"Unknown start mode: {mode}")

    def simulate_day(self) -> Dict:
        """Simulate one lunch day and return a daily summary."""
        self.day += 1
        self.current_tables = self.generate_lunch_tables()

        transmissions: List[Tuple[int, int, float]] = []
        newly_informed: Set[int] = set()

        state_at_start = dict(self.state)

        for table in self.current_tables:
            for source in table:
                if state_at_start[source] != "I":
                    continue

                # The active spreader only starts gossiping if at least one direct
                # acquaintance is sitting at the same table.
                if not self._has_direct_acquaintance_at_table(source, table):
                    continue

                # Once the source starts gossiping, any susceptible tablemate may hear it.
                for target in table:
                    if source == target:
                        continue

                    if state_at_start[target] != "S":
                        continue

                    prob = self.transmission_probability(source, target, table)

                    if prob > 0 and self.rng.random() < prob:
                        newly_informed.add(target)
                        transmissions.append((source, target, prob))

        for node in newly_informed:
            self.state[node] = "I"
            self.infection_day[node] = self.day

        if self.params.rumor_model == "SIR":
            for node, old_state in state_at_start.items():
                if old_state == "I":
                    if self.rng.random() < self.params.stifling_prob:
                        self.state[node] = "R"

        summary = self._record_history(
            newly_informed=sorted(newly_informed),
            transmissions=transmissions,
        )

        return summary

    def run(
        self,
        max_days: int = 365,
        stop_when_all_informed: bool = True,
    ) -> pd.DataFrame:
        """Run the simulation for at most max_days."""
        if not any(s in {"I", "R"} for s in self.state.values()):
            self.reset()

        for _ in range(max_days):
            if stop_when_all_informed:
                if self.n_informed_total() == self.params.n_people:
                    break

            if self.params.rumor_model == "SIR":
                if self.n_active_spreaders() == 0:
                    break

            self.simulate_day()

        return pd.DataFrame(self.history)

    def run_many(
        self,
        n_runs: int = 50,
        max_days: int = 365,
        start_mode: StartMode = "random",
    ) -> pd.DataFrame:
        """Run many independent simulations on the same friendship graph."""
        all_histories = []

        for run_id in range(n_runs):
            start_node = self.reset(start_mode=start_mode)
            hist = self.run(max_days=max_days)
            hist["run_id"] = run_id
            hist["start_node"] = start_node
            hist["start_mode"] = start_mode
            all_histories.append(hist)

        return pd.concat(all_histories, ignore_index=True)

    # ---------------------------------------------------------------------
    # Lunch seating
    # ---------------------------------------------------------------------

    def generate_lunch_tables(self) -> List[List[int]]:
        """Generate circular lunch tables for one day.

        The implementation is deliberately simple and transparent:

        - First choose who attends lunch.
        - All attendees use one shared cafeteria.
        - A preferred fixed number of tables can be set with params.n_tables.
        - If params.n_tables is None, the minimum necessary number of tables is used.
        - If params.n_tables is too small, the minimum necessary number is used and
          a warning is emitted.
        - Within each table, seats are filled sequentially, preferring direct and
          two-step acquaintances of already seated people when friend_seating_bias > 0.
        """
        attendees = [
            node
            for node in self.friendship_graph.nodes
            if self.rng.random() < self.params.lunch_attendance_prob
        ]

        if not attendees:
            return []

        return self._seat_group(
            attendees,
            preferred_n_tables=self.params.n_tables,
        )

    def _seat_group(
        self,
        people: List[int],
        preferred_n_tables: Optional[int] = None,
    ) -> List[List[int]]:
        """Seat one group of people at circular lunch tables."""
        people = list(people)
        self.rng.shuffle(people)

        if not people:
            return []

        table_size = self.params.table_size
        minimum_n_tables = math.ceil(len(people) / table_size)

        if preferred_n_tables is None:
            n_tables = minimum_n_tables

        elif preferred_n_tables < minimum_n_tables:
            warnings.warn(
                f"The requested number of tables ({preferred_n_tables}) is too small "
                f"for {len(people)} attendees with table_size={table_size}. "
                f"Using the minimum required number of tables: {minimum_n_tables}.",
                UserWarning,
            )
            n_tables = minimum_n_tables

        else:
            n_tables = preferred_n_tables

        tables: List[List[int]] = [[] for _ in range(n_tables)]
        unseated: Set[int] = set(people)

        for table in tables:
            if not unseated:
                break

            first = self.rng.choice(list(unseated))
            table.append(first)
            unseated.remove(first)

        while unseated:
            available_tables = [
                table
                for table in tables
                if len(table) < table_size
            ]

            if not available_tables:
                raise RuntimeError(
                    "No available seats left, although some people remain unseated."
                )

            table = self.rng.choice(available_tables)
            next_person = self._choose_next_seat(table, unseated)
            table.append(next_person)
            unseated.remove(next_person)

        return [table for table in tables if table]

    def _choose_next_seat(self, table: List[int], candidates: Set[int]) -> int:
        """Choose the next person to seat at a table.

        The acquaintance graph affects seating only. A candidate receives a
        seating score from the people already at the table:

        - direct acquaintance: 1.0
        - two-step acquaintance: params.seating_factor
        - otherwise: 0.0

        The final sampling weight is

            1 + friend_seating_bias * seating_score.

        Therefore, friend_seating_bias controls the strength of the social seating
        preference, while seating_factor controls how much indirect acquaintances
        matter in this preference.
        """
        bias = self.params.friend_seating_bias

        if bias <= 0:
            return self.rng.choice(list(candidates))

        cand_list = list(candidates)
        weights = []

        for cand in cand_list:
            seating_score = sum(
                self._social_seating_factor(cand, person)
                for person in table
            )

            weight = 1.0 + bias * seating_score
            weights.append(weight)

        total = sum(weights)
        probs = [w / total for w in weights]

        return self.np_rng.choice(cand_list, p=probs).item()

    def _social_seating_factor(self, a: int, b: int) -> float:
        """Return how strongly b attracts a during seating.

        This is used only by the seating algorithm, not by the transmission rule.
        """
        G = self.friendship_graph

        if G.has_edge(a, b):
            return 1.0

        a_neighbors = set(G.neighbors(a))
        b_neighbors = set(G.neighbors(b))

        if a_neighbors.intersection(b_neighbors):
            return self.params.seating_factor

        return 0.0

    # ---------------------------------------------------------------------
    # Transmission probabilities
    # ---------------------------------------------------------------------

    def transmission_probability(
        self,
        source: int,
        target: int,
        table: List[int],
    ) -> float:
        """Probability that source exposes target to the gossip at this table.

        The acquaintance graph does not directly restrict who can hear the gossip.
        It shapes seating and determines whether the source starts gossiping at the
        table. Once the source starts talking, any susceptible tablemate may hear the
        gossip. Transmission probability depends on beta and circular seating
        distance.
        """
        seat_dist = self.circular_seat_distance(source, target, table)
        distance_factor = math.exp(
            -self.params.seating_alpha * (seat_dist - 1)
        )

        prob = self.params.beta * distance_factor

        return max(0.0, min(1.0, prob))

    @staticmethod
    def circular_seat_distance(a: int, b: int, table: List[int]) -> int:
        if a not in table or b not in table:
            raise ValueError("Both people must be seated at the same table.")

        m = len(table)
        ia = table.index(a)
        ib = table.index(b)
        raw = abs(ia - ib)

        return min(raw, m - raw)
    
    def _has_direct_acquaintance_at_table(
        self,
        source: int,
        table: List[int],
    ) -> bool:
        """Return True if source has at least one direct acquaintance at the table.

        This controls whether an active spreader starts gossiping at the table.
        The gossip can still be heard by any susceptible tablemate once the source
        starts talking.
        """
        return any(
            other != source and self.friendship_graph.has_edge(source, other)
            for other in table
        )

    # ---------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------

    def n_active_spreaders(self) -> int:
        return sum(1 for s in self.state.values() if s == "I")

    def n_informed_total(self) -> int:
        return sum(1 for s in self.state.values() if s in {"I", "R"})

    def _record_history(
        self,
        newly_informed: List[int],
        transmissions: List[Tuple[int, int, float]],
    ) -> Dict:
        group_counts = {
            f"group_{group}_informed": sum(
                1
                for node, node_group in self.groups.items()
                if node_group == group and self.state[node] in {"I", "R"}
            )
            for group in sorted(set(self.groups.values()))
        }

        summary = {
            "day": self.day,
            "n_active_spreaders": self.n_active_spreaders(),
            "n_informed_total": self.n_informed_total(),
            "fraction_informed": self.n_informed_total() / self.params.n_people,
            "n_newly_informed": len(newly_informed),
            "newly_informed": newly_informed,
            "n_transmissions": len(transmissions),
            "transmissions": transmissions,
            **group_counts,
        }

        self.history.append(summary)
        return summary

    def infection_times(self) -> pd.DataFrame:
        """Return one row per person with infection day and network attributes."""
        degrees = dict(self.friendship_graph.degree())

        return pd.DataFrame(
            {
                "node": list(self.friendship_graph.nodes),
                "infection_day": [
                    self.infection_day[i]
                    for i in self.friendship_graph.nodes
                ],
                "degree": [
                    degrees[i]
                    for i in self.friendship_graph.nodes
                ],
                "group": [
                    self.groups[i]
                    for i in self.friendship_graph.nodes
                ],
                "state": [
                    self.state[i]
                    for i in self.friendship_graph.nodes
                ],
            }
        )

    def final_summary(self) -> Dict:
        """Return compact summary of the current simulation."""
        informed_days = [
            d
            for d in self.infection_day.values()
            if d is not None
        ]

        return {
            "days_elapsed": self.day,
            "n_informed_total": self.n_informed_total(),
            "fraction_informed": self.n_informed_total() / self.params.n_people,
            "all_informed": self.n_informed_total() == self.params.n_people,
            "last_infection_day": max(informed_days) if informed_days else None,
        }

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------

    def plot_friendship_graph(
        self,
        layout: str = "spring",
        show_labels: bool = True,
        title: str = "Workplace acquaintance graph",
        save_path: Optional[str] = None,
        highlight_node: Optional[int] = None,
        highlight_neighborhood: bool = False,
        base_node_size: int = 260,
    ):
        """Plot the stable workplace acquaintance graph."""
        G = self.friendship_graph
        pos = self._get_layout(layout)

        if highlight_neighborhood:
            if highlight_node is None:
                highlight_node = self._infer_initial_source()

            node_colors = self._neighborhood_highlight_colors(highlight_node)

            node_sizes = [
                base_node_size
                if node == highlight_node
                else base_node_size * 3 // 4
                if self._distance_from_focal(highlight_node, node) == 1
                else base_node_size // 2
                if self._distance_from_focal(highlight_node, node) == 2
                else base_node_size // 3
                for node in G.nodes
            ]

        else:
            node_colors = [
                self._state_color(self.state[node])
                for node in G.nodes
            ]
            node_sizes = [
                base_node_size // 3
                for _ in G.nodes
            ]

        plt.figure(figsize=(8, 6))

        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.25,
            width=0.8,
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=0.4,
        )

        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=7)

        plt.title(title)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

    def plot_current_lunch_tables(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_labels: bool = True,
        highlight_node: Optional[int] = None,
        highlight_neighborhood: bool = False,
        base_node_size: int = 260,
    ):
        """Visualize the latest lunch seating as separate circular tables."""
        if not self.current_tables:
            raise ValueError(
                "No lunch tables available. Run simulate_day() first."
            )

        if highlight_neighborhood:
            if highlight_node is None:
                highlight_node = self._infer_initial_source()

        positions = {}
        table_centers = []

        n_tables = len(self.current_tables)
        grid_cols = math.ceil(math.sqrt(n_tables))
        spacing = 3.0

        for table_idx, table in enumerate(self.current_tables):
            row = table_idx // grid_cols
            col = table_idx % grid_cols

            center_x = col * spacing
            center_y = -row * spacing

            table_centers.append((center_x, center_y))

            m = len(table)

            for seat_idx, node in enumerate(table):
                angle = 2 * math.pi * seat_idx / max(1, m)

                positions[node] = (
                    center_x + math.cos(angle),
                    center_y + math.sin(angle),
                )

        lunch_graph = nx.Graph()

        for table in self.current_tables:
            lunch_graph.add_nodes_from(table)

            if len(table) > 1:
                for idx, node in enumerate(table):
                    lunch_graph.add_edge(
                        node,
                        table[(idx + 1) % len(table)],
                    )

        if highlight_neighborhood:
            all_colors = self._neighborhood_highlight_colors(highlight_node)
            color_by_node = dict(zip(self.friendship_graph.nodes, all_colors))

            node_colors = [
                color_by_node[node]
                for node in lunch_graph.nodes
            ]

            node_sizes = [
                base_node_size
                if node == highlight_node
                else base_node_size * 3 // 4
                if self._distance_from_focal(highlight_node, node) == 1
                else base_node_size // 2
                if self._distance_from_focal(highlight_node, node) == 2
                else base_node_size // 3
                for node in lunch_graph.nodes
            ]

        else:
            node_colors = [
                self._state_color(self.state[node])
                for node in lunch_graph.nodes
            ]

            node_sizes = [
                base_node_size // 2
                for _ in lunch_graph.nodes
            ]

        plt.figure(figsize=(9, 7))

        nx.draw_networkx_edges(
            lunch_graph,
            positions,
            alpha=0.45,
            width=1.2,
        )

        nx.draw_networkx_nodes(
            lunch_graph,
            positions,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=0.4,
        )

        if show_labels:
            nx.draw_networkx_labels(lunch_graph, positions, font_size=7)

        for cx, cy in table_centers:
            circle = plt.Circle((cx, cy), 1.15, fill=False, alpha=0.25)
            plt.gca().add_patch(circle)

        plt.title(title or f"Lunch seating on day {self.day}")
        plt.axis("equal")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

    def plot_spreading_curve(
        self,
        history: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ):
        """Plot fraction informed over time."""
        if history is None:
            history = pd.DataFrame(self.history)

        if history.empty:
            raise ValueError("No history to plot. Run the model first.")

        plt.figure(figsize=(7, 4))

        plt.plot(
            history["day"],
            history["fraction_informed"],
            marker="o",
        )

        plt.xlabel("Day")
        plt.ylabel("Fraction informed")
        plt.title("Gossip spreading curve")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

    def plot_many_spreading_curves(
        self,
        histories: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """Plot average spreading curve from run_many output."""
        summary = (
            histories
            .groupby("day")["fraction_informed"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["se"] = summary["std"] / np.sqrt(summary["count"])

        plt.figure(figsize=(7, 4))

        plt.plot(
            summary["day"],
            summary["mean"],
            marker="o",
        )

        plt.fill_between(
            summary["day"],
            summary["mean"] - summary["se"],
            summary["mean"] + summary["se"],
            alpha=0.2,
        )

        plt.xlabel("Day")
        plt.ylabel("Mean fraction informed")
        plt.title("Average gossip spreading curve")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

    def _get_layout(self, layout: str) -> Dict[int, Tuple[float, float]]:
        G = self.friendship_graph

        if layout == "spring":
            return nx.spring_layout(G, seed=self.params.seed)

        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(G)

        if layout == "circular":
            return nx.circular_layout(G)

        raise ValueError(f"Unknown layout: {layout}")

    def _infer_initial_source(self) -> int:
        """Infer the initial gossip source from infection_day == 0."""
        start_nodes = [
            node
            for node, day in self.infection_day.items()
            if day == 0
        ]

        if not start_nodes:
            raise ValueError(
                "Could not infer the initial source. Call reset() first or provide "
                "highlight_node explicitly."
            )

        return start_nodes[0]

    def _distance_from_focal(
        self,
        focal_node: int,
        node: int,
    ) -> Optional[int]:
        """Return shortest-path distance from focal_node to node if it is <= 2."""
        if node == focal_node:
            return 0

        if self.friendship_graph.has_edge(focal_node, node):
            return 1

        focal_neighbors = set(self.friendship_graph.neighbors(focal_node))
        node_neighbors = set(self.friendship_graph.neighbors(node))

        if focal_neighbors.intersection(node_neighbors):
            return 2

        return None

    def _neighborhood_highlight_colors(self, focal_node: int) -> List[str]:
        """Return colors for highlighting a focal node and its 1/2-neighborhood."""
        colors = []

        for node in self.friendship_graph.nodes:
            distance = self._distance_from_focal(focal_node, node)

            if distance == 0:
                colors.append("#C40000")
            elif distance == 1:
                colors.append("#E57373")
            elif distance == 2:
                colors.append("#FFCDD2")
            else:
                colors.append("#D9D9D9")

        return colors

    @staticmethod
    def _state_color(state: str) -> str:
        """Return default colors for gossip states."""
        if state == "S":
            return "#D9D9D9"

        if state == "I":
            return "#E57373"

        if state == "R":
            return "#90A4AE"

        return "#D9D9D9"

    @staticmethod
    def _state_to_numeric_color(state: str) -> float:
        """Kept for backward compatibility with earlier versions."""
        if state == "S":
            return 0.2

        if state == "I":
            return 0.8

        if state == "R":
            return 0.5

        return 0.0