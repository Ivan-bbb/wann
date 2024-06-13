from dataclasses import dataclass


@dataclass
class Configuration:
    prob_add_conn: float = 0.25
    prob_split_conn: float = 0.2
    prob_disable_conn: float = 0.2
    prob_enable_conn: float = 0.2
    prob_change_act: float = 0.35

    dist_excess: float = 1.0
    dist_disjoint: float = 1.0

config = Configuration()
