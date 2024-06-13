from feed_forward_nn import NeuralNetwork
from population import Population
from wann import Genome
from cancer import CancerTask
from diabetes import DiabetesTask
import matplotlib.pyplot as plt
import numpy as np

def save_genome(genome: Genome, postfix: str = ""):
    nn = NeuralNetwork(genome)
    nn.save(f"./outputs/{task_name}/{generation}{postfix}")

def save_fitness_graph(champions: list[Genome]):
    g = np.arange(0, len(champions), 1)
    fitness_history = [champion.fitness for champion in champions]

    plt.plot(g, fitness_history)
    plt.grid()
    plt.savefig(f"./graphs/{task_name}")
    plt.show()


if __name__ == "__main__":
    population = Population(CancerTask)
    task_name = population.task.name
    max_generations = 3
    for generation in range(max_generations):
        population.evolve()
        print(f"[{generation}] Best fitness: {population.champions[-1].fitness}; Species: {len(population.species)}")
        if generation % 5 == 0:
            save_genome(population.champions[-1])
        
        if population.solved_at is not None or generation == max_generations - 1:
            champion = population.champions[-1]
            print(f"Solved at {population.solved_at}")
            print(f"Champion fitness {champion.fitness}")
            save_genome(champion, "_solved")
            save_fitness_graph(population.champions)

            nn = NeuralNetwork(champion)
            nn.visualize(save=True, name=task_name)
            population.task.visualize(nn)
            break
