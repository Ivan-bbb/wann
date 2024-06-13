from feed_forward_nn import NeuralNetwork
from population import Population
from wann import Genome
from cancer import CancerTask
from diabetes import DiabetesTask
from cart_pole import CartPoleTask
import matplotlib.pyplot as plt
import numpy as np
from config import config

task = CartPoleTask

def save_genome(genome: Genome, postfix: str = ""):
    nn = NeuralNetwork(genome)
    nn.save(f"./topology_genomes/{task_name}/{generation}{postfix}")

def save_fitness_graph(polulation: Population):
    g = np.arange(0, len(population.champions), 1)
    best_fitness_history = [champion.fitness for champion in population.champions]
    average_fitness_history = polulation.average_fitness

    plt.clf()

    plt.plot(g, best_fitness_history, label = "Best") 
    plt.plot(g, average_fitness_history, label = "Average") 
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.savefig(f"./graphs/{task_name}")


if __name__ == "__main__":
    population = Population(task)
    task_name = population.task.name
    max_generations = 100
    try:
        for generation in range(max_generations):
            population.evolve()
            print(f"[{generation}] Best fitness: {population.champions[-1].fitness}; Species: {len(population.species)}")
            if generation % config.save_interval == 0:
                champion = population.champions[-1]
                save_genome(champion)
                nn = NeuralNetwork(champion)
                nn.visualize(show_image=False, save=True, name=f"{task_name}/{generation}", is_solution=False)
            
            if population.solved_at is not None or generation == max_generations - 1:
                champion = population.champions[-1]
                print(f"Solved at {population.solved_at}")
                print(f"Champion fitness {champion.fitness}")
                # for i in range(len(population.champions)):
                #     print(f"genome {1}: {population.champions[i].fitness}")
                save_genome(champion, "_solved")
                save_fitness_graph(population)

                nn = NeuralNetwork(champion)
                nn.visualize(show_image=False, save=True, name=f"{task_name}/{generation}", is_solution=True)
                population.task.visualize(nn)
                break
    except KeyboardInterrupt:
        save_genome(champion)
        save_fitness_graph(population)
