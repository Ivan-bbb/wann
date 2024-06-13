from dataclasses import dataclass


@dataclass
class Configuration:
    max_depth: int = 10
    pop_size: int = 100
    target_species: int = 15
    specie_survival_rate: float = 0.3
    stagnation_age: int = 15 

    prob_add_conn: float = 0.25
    prob_split_conn: float = 0.2
    prob_disable_conn: float = 0.2
    prob_enable_conn: float = 0.2
    prob_change_act: float = 0.35

    dist_excess: float = 1.0
    dist_disjoint: float = 1.0
    comp_threshold: float = 5
    comp_threshold_delta: float = 1

    weights_pool = [-1, -0.5, 0.5, 1]
    best_eval_multiplier: float = 0.75
    avg_eval_multiplier: float = 0.25

config = Configuration()
