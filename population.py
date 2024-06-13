import random

from copy import deepcopy
from config import config
from wann import Genome, genome_distance, genome_crossover, innovation_tracker
from feed_forward_nn import NeuralNetwork
from species import Species
from base_task import Task

# турнирная выборка
def tournament_selection(genomes: list[Genome], k):
    selected = random.sample(genomes, k)
    selected.sort(key=lambda genome: genome.fitness, reverse=True)
    return selected[0]


class Population:
    def __init__(self, task):
        self.genomes: list[Genome] = []
        self.species: list[Species] = []
        self.current_compatibility_threshold: int = config.comp_threshold
        self.winners: list[Genome] = []
        self.average_fitness: list[float] = []
        self.task: Task = task()
        self.solved_id: int | None = None
        self.generation_id: int = 0
        self.input_nodes_number = self.task.input_nodes
        self.output_nodes_number = self.task.output_nodes

        self.genomes = [self.create_initial_genome() for _ in range(config.pop_size)]

    def create_initial_genome(self) -> Genome:
        genome = Genome()

        input_nodes = []
        for _ in range(self.input_nodes_number):
            input_nodes.append(genome.add_node(0))

        # bias
        genome.add_node(0)

        out_nodes = []
        for _ in range(self.output_nodes_number):
            out_nodes.append(genome.add_node(config.max_layers))

        for from_node in input_nodes:
            to_node = random.choice(out_nodes)
            genome.add_connection(from_node, to_node)

        return genome
    # распределение геномов по видам
    def speciate(self):
        for specie in self.species:
            specie.evolve_step()
       
        for genome in self.genomes:
            for specie in self.species:
                if (genome_distance(genome, specie.representative) < self.current_compatibility_threshold):
                    specie.add_genome(genome)
                    break
            else:
                new_specie = Species(representative=deepcopy(genome))
                new_specie.add_genome(genome)
                self.species.append(new_specie)
        # удаляем пустые виды
        self.species = list(filter(lambda s: len(s.genomes) > 0, self.species))
        # изменяем порог совместимости для контроля количества видов
        if len(self.species) < config.target_species:
            self.current_compatibility_threshold -= config.comp_threshold_delta
        elif len(self.species) > config.target_species:
            self.current_compatibility_threshold += config.comp_threshold_delta
        if self.current_compatibility_threshold < 0.1:
            self.current_compatibility_threshold = 0.1
    # рассчитываем приспособленность генома пир пуле весов
    def evaluate_genome(self, genome: Genome) -> float:
        eval = 0
        best_eval = 0
        best_weight = 0
        sum_eval = 0
        for weight in config.weights_pool:
            genome.set_weights(weight)
            nn = NeuralNetwork(genome)
            tmp_eval = self.task.evaluate(nn)
            sum_eval += tmp_eval
            if tmp_eval > best_eval:
                best_eval = tmp_eval
                best_weight = weight
        genome.best_weight = best_weight
        avg_eval = sum_eval / len(config.weights_pool)
        eval = best_eval * config.best_eval_multiplier + avg_eval * config.avg_eval_multiplier
        return eval
    # подсчёт приспособленности и обновление параметров видов
    def evaluate_fitness(self):
        for genome in self.genomes:
            genome.fitness = self.evaluate_genome(genome)
        for specie in self.species:
            specie.recalculate()
        self.average_fitness.append(sum(genome.fitness for genome in self.genomes) / len(self.genomes))

    def check_for_stagnation(self):
        for specie in self.species:
            specie.update_max_fitness()
            if specie.max_fitness <= specie.previous_max_fitness:
                specie.no_improvement_age += 1
            else:
                specie.no_improvement_age = 0
            specie.has_best_genome = self.winners[-1] in specie.genomes
        # удаляем не развивающиеся виды (если не имеют лучший геном)
        self.species = list(filter(lambda s: s.no_improvement_age < config.stagnation_age or s.has_best_genome, self.species))

    def find_best_genome(self):
        self.winners.append(max(self.genomes, key=lambda genome: genome.fitness))
    
    def check_for_solution(self):
        winner = self.winners[-1]
        winner.set_weights(winner.best_weight)
        nn = NeuralNetwork(winner)
        # решение есть, если хотя бы при одном весе приспособленность превышает порог, при этом общая приспособленность также должна быть выше пороговой
        if self.task.solve(nn) and winner.fitness >= self.task.threshold:
            self.solved_id = self.generation_id
    # функция для предотвращения доминации одного вида
    def reproduce_offspring(self):
        total_average = sum(specie.average_adjusted_fitness for specie in self.species)
        # определяем количество воспроизводимых особей для вида
        for specie in self.species:
            specie.offspring_number = int(round(len(self.genomes) * specie.average_adjusted_fitness / total_average))
        self.species = list(filter(lambda s: s.offspring_number > 0, self.species))
        innovation_tracker.reset()
        # заполняем виды новыми особями
        new_genomes_global = []
        for specie in self.species:
            specie.genomes.sort(key=lambda ind: ind.fitness, reverse=True)
            # оставляем определенное количество лучших особей
            keep = max(1, int(round(len(specie.genomes) * config.specie_survival_rate)))
            pool = specie.genomes[:keep]
            # добавляем лучший геном в следующее поколение без изменений
            if len(specie.genomes) >= 1:
                specie.genomes = specie.genomes[:1]
                new_genomes_global += specie.genomes
            else:
                specie.genomes = []
            # воспроизведение потомков
            while len(specie.genomes) < specie.offspring_number:
                new_genomes = []
                # если не хватает одного, то копируем лучший
                if len(pool) == 1:
                    child = deepcopy(pool[0])
                    child.mutate()
                    new_genomes.append(child)
                # скрещивание и мутация
                else:
                    parent1 = deepcopy(tournament_selection(pool, min(len(pool), 3)))
                    parent2 = deepcopy(tournament_selection(pool, min(len(pool), 3)))
                    child = genome_crossover(parent1, parent2)
                    child.mutate()
                    new_genomes.append(child)
                specie.genomes += new_genomes
                new_genomes_global += new_genomes
        self.genomes = new_genomes_global

    def evolve(self):
        self.speciate()
        self.evaluate_fitness()
        self.find_best_genome()
        self.check_for_solution()
        self.check_for_stagnation()
        self.reproduce_offspring()
        self.generation_id += 1
