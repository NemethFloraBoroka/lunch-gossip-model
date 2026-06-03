# -------------------------------------------------------------------------
# Experiment helper functions for the exposure-based Lunch Gossip Model
# -------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300

from codebase.GossipModel import LunchGossipModel, LunchGossipParams, StartMode


# -------------------------------------------------------------------------
# General utility functions
# -------------------------------------------------------------------------

def has_common_neighbor(G: nx.Graph, source: int, target: int) -> bool:
    """Return True if two nodes have at least one common neighbor."""
    source_neighbors = set(G.neighbors(source))
    target_neighbors = set(G.neighbors(target))
    return bool(source_neighbors.intersection(target_neighbors))


def pad_history_to_max_days(history: pd.DataFrame, max_days: int) -> pd.DataFrame:
    """Pad a simulation history to have one row for every day.

    If a simulation stops before max_days, the last observed values are carried
    forward. This makes histories comparable and averageable across runs.

    Parameters
    ----------
    history:
        Simulation history with a ``day`` column.
    max_days:
        Maximum simulation day.

    Returns
    -------
    pd.DataFrame
        History with days 0, 1, ..., max_days.
    """
    full_days = pd.DataFrame({"day": np.arange(0, max_days + 1)})

    padded = full_days.merge(history, on="day", how="left")

    columns_to_fill = [
        col for col in padded.columns
        if col != "day"
    ]

    for col in columns_to_fill:
        padded[col] = padded[col].ffill()

    return padded


def compact_spreading_history(
    history: pd.DataFrame,
    include_active_spreaders: bool = False,
) -> pd.DataFrame:
    """Keep only the spreading columns needed for the experiments."""
    columns = [
        "day",
        "n_informed_total",
        "fraction_informed",
    ]

    if include_active_spreaders and "n_active_spreaders" in history.columns:
        columns.append("n_active_spreaders")

    return history[columns].copy()


def time_to_fraction(
    history: pd.DataFrame,
    threshold: float = 0.8,
    max_days: Optional[int] = None,
) -> float:
    """Return the first day when fraction_informed reaches a threshold.

    If the threshold is not reached, returns np.nan by default. If max_days is
    provided, returns max_days + 1 instead. This is useful for boxplots, because
    failed runs remain visible as values just beyond the simulation horizon.
    """
    reached = history.loc[
        history["fraction_informed"] >= threshold,
        "day",
    ]

    if not reached.empty:
        return float(reached.iloc[0])

    if max_days is not None:
        return float(max_days + 1)

    return np.nan


# -------------------------------------------------------------------------
# Summary functions
# -------------------------------------------------------------------------

def summarize_spreading_curves(histories: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard error for informed-fraction curves.

    The input must contain columns:
        model, day, fraction_informed
    """
    summary = (
        histories
        .groupby(["model", "day"])["fraction_informed"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["lower"] = summary["mean"] - summary["se"]
    summary["upper"] = summary["mean"] + summary["se"]

    return summary


def summarize_active_spreader_curves(histories: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard error for active-spreader curves.

    The input must contain columns:
        model, day, n_active_spreaders
    """
    summary = (
        histories
        .groupby(["model", "day"])["n_active_spreaders"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["lower"] = summary["mean"] - summary["se"]
    summary["upper"] = summary["mean"] + summary["se"]

    return summary


# -------------------------------------------------------------------------
# General plotting functions
# -------------------------------------------------------------------------

def plot_static_vs_lunch_average_curves(
    summary: pd.DataFrame,
    show_se_band: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = "Static acquaintance spreading vs. temporal lunch-contact spreading",
    colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (6, 3.5),
):
    """Plot average informed-fraction curves.

    Despite the function name, this is now a general line-plot function. It can
    be used for static-vs-lunch comparison, start-node comparison, table-size
    comparison, seating-bias comparison, and SI/SIR comparison, as long as the
    summary dataframe contains:
        model, day, mean, lower, upper
    """
    plt.figure(figsize=figsize)

    if colors is None:
        colors = {}

    for model_name, group in summary.groupby("model"):
        group = group.sort_values("day")
        color = colors.get(model_name, None)

        plt.plot(
            group["day"],
            group["mean"],
            marker="o",
            markersize=4,
            linewidth=1,
            label=model_name,
            color=color,
        )

        if show_se_band:
            plt.fill_between(
                group["day"],
                group["lower"],
                group["upper"],
                alpha=0.18,
                color=color,
            )

    plt.xlabel("Day", fontsize=14)
    plt.ylabel("|Fraction informed|", fontsize=14)
    plt.title(title, fontsize=15)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend(frameon=True, fontsize=12, loc="lower right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_active_spreader_curves(
    summary: pd.DataFrame,
    show_se_band: bool = True,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    title: str = "Active gossip spreaders over time",
    figsize: Tuple[float, float] = (6, 3.5),
):
    """Plot average number of active spreaders over time."""
    if colors is None:
        colors = {}

    plt.figure(figsize=figsize)

    for model_name, group in summary.groupby("model"):
        group = group.sort_values("day")
        color = colors.get(model_name, None)

        plt.plot(
            group["day"],
            group["mean"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=model_name,
            color=color,
        )

        if show_se_band:
            plt.fill_between(
                group["day"],
                group["lower"],
                group["upper"],
                alpha=0.18,
                color=color,
            )

    plt.xlabel("Day", fontsize=14)
    plt.ylabel("|Number of active spreaders|", fontsize=14)
    plt.title(title, fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend(frameon=True, fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_threshold_boxplot(
    data_frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    order: List,
    labels: List[str],
    colors: Optional[Dict] = None,
    threshold: float = 0.8,
    xlabel: str = "",
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4.5),
):
    """Generic colored boxplot for time-to-threshold experiments."""
    if colors is None:
        colors = {}

    data = [
        data_frame.loc[
            data_frame[group_column] == value,
            value_column,
        ].dropna().values
        for value in order
    ]

    fig, ax = plt.subplots(figsize=figsize)

    box = ax.boxplot(
        data,
        labels=labels,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
    )

    for patch, value in zip(box["boxes"], order):
        patch.set_facecolor(colors.get(value, "#D9D9D9"))
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(f"Days to {int(threshold * 100)}% informed", fontsize=15)
    ax.set_title(title, fontsize=17)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0, ha="center")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -------------------------------------------------------------------------
# Network characterization
# -------------------------------------------------------------------------

def compute_network_summary(model: LunchGossipModel) -> pd.DataFrame:
    """Compute basic summary statistics for the acquaintance graph."""
    G = model.friendship_graph

    degrees = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)

    if nx.is_connected(G):
        average_shortest_path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
        largest_component_fraction = 1.0
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()
        average_shortest_path_length = nx.average_shortest_path_length(G_lcc)
        diameter = nx.diameter(G_lcc)
        largest_component_fraction = len(largest_cc) / G.number_of_nodes()

    summary = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "average_degree": np.mean(list(degrees.values())),
        "median_degree": np.median(list(degrees.values())),
        "max_degree": np.max(list(degrees.values())),
        "average_betweenness": np.mean(list(betweenness.values())),
        "median_betweenness": np.median(list(betweenness.values())),
        "max_betweenness": np.max(list(betweenness.values())),
        "average_clustering": np.mean(list(clustering.values())),
        "largest_component_fraction": largest_component_fraction,
        "average_shortest_path_length_lcc": average_shortest_path_length,
        "diameter_lcc": diameter,
    }

    return pd.DataFrame([summary])


def run_degree_distribution_ensemble(
    base_params: LunchGossipParams,
    n_runs: int = 300,
    seed_start: int = 5000,
) -> pd.DataFrame:
    """Generate many acquaintance graphs and collect their degree counts.

    Returns one row per degree value and run:
        run_id, degree, count, fraction
    """
    all_rows = []

    for run_id in range(n_runs):
        run_seed = seed_start + run_id
        run_params = replace(base_params, seed=run_seed)
        model = LunchGossipModel(run_params)

        degrees = np.array([
            degree for _, degree in model.friendship_graph.degree()
        ])

        unique_degrees, counts = np.unique(degrees, return_counts=True)

        for degree, count in zip(unique_degrees, counts):
            all_rows.append(
                {
                    "run_id": run_id,
                    "degree": int(degree),
                    "count": int(count),
                    "fraction": count / base_params.n_people,
                }
            )

    return pd.DataFrame(all_rows)


def summarize_degree_distribution_ensemble(
    degree_runs: pd.DataFrame,
) -> pd.DataFrame:
    """Average degree distributions over many generated graphs."""
    max_degree = int(degree_runs["degree"].max())

    full_index = pd.MultiIndex.from_product(
        [
            sorted(degree_runs["run_id"].unique()),
            np.arange(0, max_degree + 1),
        ],
        names=["run_id", "degree"],
    )

    completed = (
        degree_runs
        .set_index(["run_id", "degree"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    n_nodes_per_run = (
        completed
        .groupby("run_id")["count"]
        .sum()
        .rename("n_nodes")
        .reset_index()
    )

    completed = completed.merge(n_nodes_per_run, on="run_id", how="left")
    completed["fraction"] = completed["count"] / completed["n_nodes"]

    summary = (
        completed
        .groupby("degree")["fraction"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary["std"] = summary["std"].fillna(0.0)
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["lower"] = summary["mean"] - summary["se"]
    summary["upper"] = summary["mean"] + summary["se"]

    return summary


def plot_average_degree_distribution(
    degree_summary: pd.DataFrame,
    color: str = "#4C72B0",
    show_se_band: bool = True,
    save_path: Optional[str] = None,
    title: str = "Average degree distribution of the acquaintance graph",
    figsize: Tuple[float, float] = (6, 3.5),
):
    """Plot the average degree distribution across many generated graphs."""
    degree_summary = degree_summary.sort_values("degree")

    plt.figure(figsize=figsize)

    plt.bar(
        degree_summary["degree"],
        degree_summary["mean"],
        color=color,
        edgecolor="black",
        alpha=0.8,
        label="Mean fraction of nodes",
    )

    if show_se_band:
        plt.errorbar(
            degree_summary["degree"],
            degree_summary["mean"],
            yerr=degree_summary["se"],
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=2,
            alpha=0.8,
        )

    plt.xlabel("Degree", fontsize=14)
    plt.ylabel("Mean fraction of nodes", fontsize=14)
    plt.title(title, fontsize=15)
    plt.grid(axis="y", alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -------------------------------------------------------------------------
# Static baseline vs temporal lunch-contact model
# -------------------------------------------------------------------------

def simulate_static_acquaintance_spreading(
    model: LunchGossipModel,
    max_days: int = 60,
    include_two_step: bool = True,
) -> pd.DataFrame:
    """Simulate a static acquaintance-graph spreading baseline.

    This is a classical reference model. It is not the same mechanism as the
    updated exposure-based lunch model.

    In the exposure-based lunch model:
        acquaintance graph -> shapes seating
        lunch table        -> creates exposure
        any tablemate      -> may hear the gossip

    In this static baseline:
        gossip spreads directly through the acquaintance graph.

    If include_two_step is True, two-step acquaintances can also transmit with
    reduced probability, weighted by params.seating_factor. Here, seating_factor
    is reused as a reduced social-proximity weight for the static reference
    model.
    """
    history = []

    for day in range(max_days + 1):
        n_informed = model.n_informed_total()

        history.append(
            {
                "day": day,
                "n_informed_total": n_informed,
                "fraction_informed": n_informed / model.params.n_people,
            }
        )

        if n_informed == model.params.n_people:
            break

        state_at_start = dict(model.state)
        newly_informed = set()

        informed_nodes = [
            node for node, state in state_at_start.items()
            if state == "I"
        ]

        susceptible_nodes = [
            node for node, state in state_at_start.items()
            if state == "S"
        ]

        for source in informed_nodes:
            for target in susceptible_nodes:
                if model.friendship_graph.has_edge(source, target):
                    social_factor = 1.0

                elif include_two_step and has_common_neighbor(
                    model.friendship_graph,
                    source,
                    target,
                ):
                    social_factor = model.params.seating_factor

                else:
                    social_factor = 0.0

                if social_factor == 0:
                    continue

                transmission_prob = min(
                    1.0,
                    model.params.beta * social_factor,
                )

                if model.rng.random() < transmission_prob:
                    newly_informed.add(target)

        for node in newly_informed:
            model.state[node] = "I"
            model.infection_day[node] = day + 1

    return pd.DataFrame(history)


def run_static_vs_lunch_comparison(
    base_params: LunchGossipParams,
    n_runs: int = 200,
    max_days: int = 60,
    start_mode: StartMode = "highest_degree",
    include_two_step_static: bool = True,
    seed_start: int = 1000,
) -> pd.DataFrame:
    """Compare static acquaintance spreading to temporal lunch-contact spreading.

    For each run, the same random seed, the same generated acquaintance graph,
    and the same starting node are used for both mechanisms.
    """
    all_histories = []

    for run_id in range(n_runs):
        run_seed = seed_start + run_id
        run_params = replace(base_params, seed=run_seed)

        # Static reference model.
        static_model = LunchGossipModel(run_params)
        start_node = static_model.reset(start_mode=start_mode)

        static_history = simulate_static_acquaintance_spreading(
            static_model,
            max_days=max_days,
            include_two_step=include_two_step_static,
        )

        static_history = pad_history_to_max_days(static_history, max_days)
        static_history["run_id"] = run_id
        static_history["model"] = "Static acquaintance graph"

        # Temporal lunch-contact model.
        lunch_model = LunchGossipModel(run_params)
        lunch_model.reset(start_node=start_node)

        lunch_history = lunch_model.run(
            max_days=max_days,
            stop_when_all_informed=True,
        )

        lunch_history = compact_spreading_history(lunch_history)
        lunch_history = pad_history_to_max_days(lunch_history, max_days)
        lunch_history["run_id"] = run_id
        lunch_history["model"] = "Temporal lunch-contact model"

        all_histories.append(static_history)
        all_histories.append(lunch_history)

    return pd.concat(all_histories, ignore_index=True)


# -------------------------------------------------------------------------
# Initial source node experiment
# -------------------------------------------------------------------------

def run_start_node_comparison(
    base_params: LunchGossipParams,
    start_modes: Optional[List[StartMode]] = None,
    n_runs: int = 200,
    max_days: int = 60,
    threshold: float = 0.8,
    seed_start: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare gossip spreading for different initial source strategies."""
    if start_modes is None:
        start_modes = [
            "random",
            "highest_degree",
            "lowest_degree",
            "highest_betweenness",
        ]

    all_histories = []
    threshold_rows = []

    for run_id in range(n_runs):
        run_seed = seed_start + run_id
        run_params = replace(base_params, seed=run_seed)

        for start_mode in start_modes:
            model = LunchGossipModel(run_params)
            start_node = model.reset(start_mode=start_mode)

            history = model.run(
                max_days=max_days,
                stop_when_all_informed=True,
            )

            compact_history = compact_spreading_history(history)
            compact_history = pad_history_to_max_days(
                compact_history,
                max_days,
            )

            compact_history["run_id"] = run_id
            compact_history["start_mode"] = start_mode
            compact_history["start_node"] = start_node
            compact_history["model"] = start_mode

            all_histories.append(compact_history)

            threshold_rows.append(
                {
                    "run_id": run_id,
                    "start_mode": start_mode,
                    "start_node": start_node,
                    "threshold": threshold,
                    "time_to_threshold": time_to_fraction(
                        compact_history,
                        threshold=threshold,
                        max_days=max_days,
                    ),
                    "final_fraction_informed": compact_history[
                        "fraction_informed"
                    ].iloc[-1],
                }
            )

    histories = pd.concat(all_histories, ignore_index=True)
    t_thresholds = pd.DataFrame(threshold_rows)

    return histories, t_thresholds


def plot_start_node_threshold_boxplot(
    t_thresholds: pd.DataFrame,
    threshold: float = 0.8,
    order: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4.5),
    xlabel: str = "Initial source selection",
):
    """Plot boxplots of time-to-threshold for start-node strategies."""
    if order is None:
        order = [
            "random",
            "highest_degree",
            "lowest_degree",
            "highest_betweenness",
        ]

    labels = [mode.replace("_", "\n") for mode in order]

    plot_threshold_boxplot(
        data_frame=t_thresholds,
        value_column="time_to_threshold",
        group_column="start_mode",
        order=order,
        labels=labels,
        colors=colors,
        threshold=threshold,
        xlabel=xlabel,
        title=title or "Effect of initial source node",
        save_path=save_path,
        figsize=figsize,
    )


# -------------------------------------------------------------------------
# Table size experiment with fixed total number of chairs
# -------------------------------------------------------------------------

def run_table_size_comparison(
    base_params: LunchGossipParams,
    table_sizes: List[int],
    total_chairs: int,
    n_runs: int = 200,
    max_days: int = 60,
    threshold: float = 0.8,
    start_mode: StartMode = "highest_degree",
    seed_start: int = 3000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare table sizes while keeping total seating capacity fixed."""
    invalid_sizes = [
        size for size in table_sizes
        if total_chairs % size != 0
    ]

    if invalid_sizes:
        raise ValueError(
            f"Each table size must divide total_chairs exactly. "
            f"Invalid table sizes: {invalid_sizes}. "
            f"total_chairs={total_chairs}."
        )

    all_histories = []
    threshold_rows = []

    for run_id in range(n_runs):
        run_seed = seed_start + run_id

        reference_params = replace(base_params, seed=run_seed)
        reference_model = LunchGossipModel(reference_params)
        start_node = reference_model.reset(start_mode=start_mode)

        for table_size in table_sizes:
            n_tables = total_chairs // table_size

            run_params = replace(
                base_params,
                seed=run_seed,
                table_size=table_size,
                n_tables=n_tables,
            )

            model = LunchGossipModel(run_params)
            model.reset(start_node=start_node)

            history = model.run(
                max_days=max_days,
                stop_when_all_informed=True,
            )

            compact_history = compact_spreading_history(history)
            compact_history = pad_history_to_max_days(
                compact_history,
                max_days,
            )

            compact_history["run_id"] = run_id
            compact_history["table_size"] = table_size
            compact_history["n_tables"] = n_tables
            compact_history["total_chairs"] = total_chairs
            compact_history["start_node"] = start_node
            compact_history["model"] = f"table size = {table_size}"

            all_histories.append(compact_history)

            threshold_rows.append(
                {
                    "run_id": run_id,
                    "table_size": table_size,
                    "n_tables": n_tables,
                    "total_chairs": total_chairs,
                    "start_node": start_node,
                    "threshold": threshold,
                    "time_to_threshold": time_to_fraction(
                        compact_history,
                        threshold=threshold,
                        max_days=max_days,
                    ),
                    "final_fraction_informed": compact_history[
                        "fraction_informed"
                    ].iloc[-1],
                }
            )

    histories = pd.concat(all_histories, ignore_index=True)
    t_thresholds = pd.DataFrame(threshold_rows)

    return histories, t_thresholds


def plot_table_size_threshold_boxplot(
    t_thresholds: pd.DataFrame,
    threshold: float = 0.8,
    order: Optional[List[int]] = None,
    colors: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4.5),
    xlabel: str = "Table size",
):
    """Plot boxplots of time-to-threshold for table sizes."""
    if order is None:
        order = sorted(t_thresholds["table_size"].unique())

    labels = [
        (
            f"{table_size} seats\n"
            f"({int(t_thresholds.loc[t_thresholds['table_size'] == table_size, 'n_tables'].iloc[0])} tables)"
        )
        for table_size in order
    ]

    plot_threshold_boxplot(
        data_frame=t_thresholds,
        value_column="time_to_threshold",
        group_column="table_size",
        order=order,
        labels=labels,
        colors=colors,
        threshold=threshold,
        xlabel=xlabel,
        title=title or "Effect of table size",
        save_path=save_path,
        figsize=figsize,
    )


# -------------------------------------------------------------------------
# Friend seating bias experiment
# -------------------------------------------------------------------------

def run_friend_seating_bias_comparison(
    base_params: LunchGossipParams,
    bias_values: List[float],
    n_runs: int = 200,
    max_days: int = 60,
    threshold: float = 0.8,
    start_mode: StartMode = "highest_degree",
    seed_start: int = 4000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare different friend-seating-bias values.

    In the updated model, friend_seating_bias affects how strongly the
    acquaintance graph shapes lunch seating. It does not directly modify the
    transmission probability.
    """
    all_histories = []
    threshold_rows = []

    for run_id in range(n_runs):
        run_seed = seed_start + run_id

        reference_params = replace(base_params, seed=run_seed)
        reference_model = LunchGossipModel(reference_params)
        start_node = reference_model.reset(start_mode=start_mode)

        for bias in bias_values:
            run_params = replace(
                base_params,
                seed=run_seed,
                friend_seating_bias=bias,
            )

            model = LunchGossipModel(run_params)
            model.reset(start_node=start_node)

            history = model.run(
                max_days=max_days,
                stop_when_all_informed=True,
            )

            compact_history = compact_spreading_history(history)
            compact_history = pad_history_to_max_days(
                compact_history,
                max_days,
            )

            compact_history["run_id"] = run_id
            compact_history["friend_seating_bias"] = bias
            compact_history["start_node"] = start_node
            compact_history["model"] = f"bias = {bias:g}"

            all_histories.append(compact_history)

            threshold_rows.append(
                {
                    "run_id": run_id,
                    "friend_seating_bias": bias,
                    "start_node": start_node,
                    "threshold": threshold,
                    "time_to_threshold": time_to_fraction(
                        compact_history,
                        threshold=threshold,
                        max_days=max_days,
                    ),
                    "final_fraction_informed": compact_history[
                        "fraction_informed"
                    ].iloc[-1],
                }
            )

    histories = pd.concat(all_histories, ignore_index=True)
    t_thresholds = pd.DataFrame(threshold_rows)

    return histories, t_thresholds


def plot_friend_seating_bias_threshold_boxplot(
    t_thresholds: pd.DataFrame,
    threshold: float = 0.8,
    order: Optional[List[float]] = None,
    colors: Optional[Dict[float, str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4.5),
    xlabel: str = "Friend seating bias",
):
    """Plot boxplots of time-to-threshold for seating-bias values."""
    if order is None:
        order = sorted(t_thresholds["friend_seating_bias"].unique())

    labels = [f"b = {bias:g}" for bias in order]

    plot_threshold_boxplot(
        data_frame=t_thresholds,
        value_column="time_to_threshold",
        group_column="friend_seating_bias",
        order=order,
        labels=labels,
        colors=colors,
        threshold=threshold,
        xlabel=xlabel,
        title=title or "Effect of friend seating bias",
        save_path=save_path,
        figsize=figsize,
    )


# -------------------------------------------------------------------------
# SIR stifling experiment
# -------------------------------------------------------------------------

def run_sir_stifling_comparison(
    base_params: LunchGossipParams,
    stifling_prob_values: List[float],
    n_runs: int = 200,
    max_days: int = 60,
    threshold: float = 0.8,
    start_mode: StartMode = "highest_degree",
    seed_start: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare SI and SIR gossip spreading for stifling probabilities.

    A stifling probability of 0.0 is treated as the SI baseline.
    Positive values are simulated with the SIR model.
    """
    all_histories = []
    threshold_rows = []

    for run_id in range(n_runs):
        run_seed = seed_start + run_id

        reference_params = replace(base_params, seed=run_seed)
        reference_model = LunchGossipModel(reference_params)
        start_node = reference_model.reset(start_mode=start_mode)

        for stifling_prob in stifling_prob_values:
            if stifling_prob == 0:
                rumor_model = "SI"
                model_label = "SI"
            else:
                rumor_model = "SIR"
                model_label = f"SIR, rho = {stifling_prob:g}"

            run_params = replace(
                base_params,
                seed=run_seed,
                rumor_model=rumor_model,
                stifling_prob=stifling_prob,
            )

            model = LunchGossipModel(run_params)
            model.reset(start_node=start_node)

            history = model.run(
                max_days=max_days,
                stop_when_all_informed=True,
            )

            compact_history = compact_spreading_history(
                history,
                include_active_spreaders=True,
            )

            compact_history = pad_history_to_max_days(
                compact_history,
                max_days,
            )

            compact_history["run_id"] = run_id
            compact_history["rumor_model"] = rumor_model
            compact_history["stifling_prob"] = stifling_prob
            compact_history["start_node"] = start_node
            compact_history["model"] = model_label

            all_histories.append(compact_history)

            threshold_rows.append(
                {
                    "run_id": run_id,
                    "rumor_model": rumor_model,
                    "stifling_prob": stifling_prob,
                    "model": model_label,
                    "start_node": start_node,
                    "threshold": threshold,
                    "time_to_threshold": time_to_fraction(
                        compact_history,
                        threshold=threshold,
                        max_days=max_days,
                    ),
                    "final_fraction_informed": compact_history[
                        "fraction_informed"
                    ].iloc[-1],
                    "final_active_spreaders": compact_history[
                        "n_active_spreaders"
                    ].iloc[-1],
                }
            )

    histories = pd.concat(all_histories, ignore_index=True)
    t_thresholds = pd.DataFrame(threshold_rows)

    return histories, t_thresholds


def plot_sir_stifling_threshold_boxplot(
    t_thresholds: pd.DataFrame,
    threshold: float = 0.8,
    order: Optional[List[float]] = None,
    colors: Optional[Dict[float, str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7, 4.5),
    xlabel: str = "Rumour model and stifling probability",
):
    """Plot boxplots of time-to-threshold for SIR stifling probabilities."""
    if order is None:
        order = sorted(t_thresholds["stifling_prob"].unique())

    labels = [
        "SI" if stifling_prob == 0 else f"SIR\nrho = {stifling_prob:g}"
        for stifling_prob in order
    ]

    plot_threshold_boxplot(
        data_frame=t_thresholds,
        value_column="time_to_threshold",
        group_column="stifling_prob",
        order=order,
        labels=labels,
        colors=colors,
        threshold=threshold,
        xlabel=xlabel,
        title=title or f"Effect of SIR stifling on T{int(threshold * 100)}",
        save_path=save_path,
        figsize=figsize,
    )