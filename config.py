from dataclasses import dataclass


@dataclass
class Configuration:
    save_interval: int = 5                  # интервал сохранения нейросети

    max_layers: int = 10                    # максимальная количество слоёв
    pop_size: int = 100                     # размер популяции
    target_species: int = 15                # целевое количество видов
    specie_survival_rate: float = 0.3       # коэффициент выживаемости вида
    stagnation_age: int = 15                # возраст стагнации

    prob_add_conn: float = 0.25             # вероятность добавления связи
    prob_add_node: float = 0.2              # вероятность разделения нейрона
    prob_disable_conn: float = 0.2          # вероятность отключения связи
    prob_enable_conn: float = 0.2           # вероятность включения связи
    prob_change_act: float = 0.35           # вероятность изменения функции активации

    dist_disjoint: float = 1.0              # расстояния для разобщённых генов
    dist_excess: float = 1.0                # расстояние для избыточных генов
    comp_threshold: float = 5               # порог совместимости
    comp_threshold_delta: float = 1         # изменение порога совместимости

    weights_pool = [-1, -0.5, 0.5, 1]       # пул весов
    best_eval_multiplier: float = 0.75      # множитель для лучшего результата из пула
    avg_eval_multiplier: float = 0.25       # множитель для среднего результата из пула

config = Configuration()
