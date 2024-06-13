from wann import Genome


class Species:
    def __init__(self, representative):
        self.representative: Genome = representative
        self.genomes: list[Genome] = []
        self.max_fitness: float = 0
        self.previous_max_fitness: float = 0
        self.average_adjusted_fitness: float = 0
        self.age: int = 0
        self.no_improvement_age: int = 0
        self.has_best_genome: bool = False
        self.offspring_number: int = 0

    def add_genome(self, genome: Genome):
        self.genomes.append(genome)
        self.update_adjusted_fitness()
        self.average_adjusted_fitness = sum([g.adjusted_fitness for g in self.genomes]) / len(self.genomes)

    def update_adjusted_fitness(self):
        for genome in self.genomes:
            genome.adjusted_fitness = genome.fitness / len(self.genomes)

    def recalculate(self):
        self.update_adjusted_fitness()
        self.average_adjusted_fitness = sum([g.adjusted_fitness for g in self.genomes]) / len(self.genomes)

    def update_max_fitness(self):
        self.previous_max_fitness = self.max_fitness
        self.max_fitness = max(self.genomes, key=lambda genome: genome.fitness).fitness

    def evolve_step(self):
        self.genomes = []
        self.average_adjusted_fitness = 0
        self.age += 1
