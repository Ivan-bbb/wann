import random

from copy import deepcopy
from config import config
from wann import Genome, genome_distance, genome_crossover, innovation_tracker
from feed_forward_nn import NeuralNetwork
from species import Species
from base_task import Task


def tournament_selection(genomes: list[Genome], k):
    selected = random.sample(genomes, k)
    selected.sort(key=lambda genome: genome.fitness, reverse=True)
    return selected[0]


class Population:
    def __init__(self, task):
        self.genomes: list[Genome] = []
        self.species: list[Species] = []
        self.current_compatibility_threshold: int = config.comp_threshold
        self.champions: list[Genome] = []
        self.task: Task = task()
        self.solved_at: int | None = None
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
        input_nodes.append(genome.add_node(0))

        out_nodes = []

        for _ in range(self.output_nodes_number):
            out_nodes.append(genome.add_node(config.max_depth + 1))

        for from_node in input_nodes:
            to_node = random.choice(out_nodes)
            genome.add_connection(from_node, to_node, 1.0)

        return genome

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

        self.species = list(filter(lambda s: len(s.genomes) > 0, self.species))

        if len(self.species) < config.target_species:
            self.current_compatibility_threshold -= config.comp_threshold_delta
        elif len(self.species) > config.target_species:
            self.current_compatibility_threshold += config.comp_threshold_delta
        if self.current_compatibility_threshold < 0.1:
            self.current_compatibility_threshold = 0.1

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

    def evaluate_fitness(self):
        for genome in self.genomes:
            genome.fitness = self.evaluate_genome(genome)
        for specie in self.species:
            specie.recalculate()

    def check_for_stagnation(self):
        for specie in self.species:
            specie.update_max_fitness()
            if specie.max_fitness <= specie.previous_max_fitness:
                specie.no_improvement_age += 1
            else:
                specie.no_improvement_age = 0
            specie.has_best_genome = self.champions[-1] in specie.genomes

        self.species = list(filter(lambda s: s.no_improvement_age < config.stagnation_age or s.has_best_genome, self.species))

    def find_best_genome(self):
        self.champions.append(max(self.genomes, key=lambda genome: genome.fitness))

    def look_for_solution(self):
        champion = self.champions[-1]
        champion.set_weights(champion.best_weight)
        nn = NeuralNetwork(champion)
        
        if self.task.solve(nn):
            self.solved_at = self.generation_id

    def reproduce_offspring(self):
        total_average = sum(specie.average_adjusted_fitness for specie in self.species)
        for specie in self.species:
            specie.offspring_number = int(round(len(self.genomes) * specie.average_adjusted_fitness / total_average))
        self.species = list(filter(lambda s: s.offspring_number > 0, self.species))
        innovation_tracker.reset()

        new_genomes_global = []
        for specie in self.species:
            specie.genomes.sort(key=lambda ind: ind.fitness, reverse=True)
            keep = max(1, int(round(len(specie.genomes) * config.specie_survival_rate)))
            pool = specie.genomes[:keep]
            if len(specie.genomes) >= 1:
                specie.genomes = specie.genomes[:1]
                new_genomes_global += specie.genomes
            else:
                specie.genomes = []

            while len(specie.genomes) < specie.offspring_number:
                new_genomes = []
                if len(pool) == 1:
                    child = deepcopy(pool[0])
                    child.mutate()
                    new_genomes.append(child)
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
        print(f"genomes: {len(self.genomes)}; species: {len(self.species)}")
        self.speciate()
        self.evaluate_fitness()
        self.find_best_genome()
        self.look_for_solution()
        self.check_for_stagnation()
        self.reproduce_offspring()
        self.generation_id += 1
