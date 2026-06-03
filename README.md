# Lunch Gossip Model

A temporal lunch-contact model for simulating workplace gossip spreading on acquaintance networks.

A detailed description of the model, its motivation, mathematical formulation, simulation experiments, and example results is available in [`Gossip_model.pdf`](Gossip_model.pdf).

This repository contains a small stochastic simulation model developed for a network science course assignment. The model represents a workplace as a stable acquaintance network and simulates gossip spreading through daily lunch-table encounters.

The aim is not to predict real workplace behaviour, but to provide a transparent toy model that can be modified, visualized, and used for exploring different spreading scenarios.

## Core idea

The model separates three mechanisms:

1. **Stable acquaintance network**  
   Employees are represented as nodes, and acquaintance relations are represented as edges.

2. **Daily lunch seating**  
   On each simulated day, employees are assigned to circular lunch tables. The seating process is stochastic but biased: employees are more likely to sit near direct or indirect acquaintances.

3. **Lunch-table gossip exposure**  
   An active spreader only starts gossiping at a table if at least one direct acquaintance is present. Once the gossip is mentioned, any susceptible tablemate may hear it, with a probability that decreases with circular seating distance.

This makes the model different from a purely static spreading model. The acquaintance graph shapes social opportunity, while the lunch table creates temporary exposure situations.

## Model features

The implementation supports:

- generation of workplace acquaintance networks;
- stochastic block model groups/departments;
- daily randomized lunch seating;
- socially biased seating based on direct and two-step acquaintances;
- circular table layouts;
- SI and SIR-like gossip dynamics;
- different initial source-node strategies;
- static acquaintance-graph baseline comparison;
- repeated stochastic simulations;
- average spreading curves;
- degree distribution summaries;
- time-to-threshold analysis, such as \(T_{80}\);
- boxplots for comparing parameter settings;
- visualization of acquaintance networks and lunch seating configurations.

## Main model parameters

| Parameter | Meaning |
|---|---|
| `n_people` | Number of employees |
| `graph_type` | Type of acquaintance graph |
| `n_groups` | Number of workplace groups in the stochastic block model |
| `p_intra` | Within-group edge probability |
| `p_inter` | Between-group edge probability |
| `table_size` | Maximum number of people at one lunch table |
| `n_tables` | Preferred number of available lunch tables |
| `beta` | Baseline gossip transmission probability |
| `seating_factor` | Weight of two-step acquaintances in the seating process |
| `seating_alpha` | Strength of distance decay around circular tables |
| `friend_seating_bias` | Strength of preference for sitting near acquaintances |
| `lunch_attendance_prob` | Probability that an employee attends lunch on a given day |
| `rumor_model` | `"SI"` or `"SIR"` |
| `stifling_prob` | Daily probability that an active spreader becomes inactive in SIR mode |

## Example usage

```python
from codebase.GossipModel import *
from codebase.experiments import *

params = LunchGossipParams(
    n_people=50,
    avg_degree=3,
    table_size=6,
    n_tables=12,
    beta=0.30,
    seating_factor=0.25,
    seating_alpha=0.70,
    friend_seating_bias=3.0,
    lunch_attendance_prob=1.0,
    graph_type="sbm",
    n_groups=3,
    p_intra=0.30,
    p_inter=0.03,
    seed=20,
)

model = LunchGossipModel(params)

start_node = model.reset(start_mode="highest_degree")
history = model.run(max_days=60)

model.plot_friendship_graph(
    highlight_node=start_node,
    highlight_neighborhood=True,
)

model.plot_spreading_curve(history)
