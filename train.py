from feed_forward_nn import NeuralNetwork
from population import Population
from wann import Genome
from cancer import CancerTask
from diabetes import DiabetesTask
from cart_pole import CartPoleTask
import matplotlib.pyplot as plt
import numpy as np
from config import config

task = CancerTask

def save_genome(genome: Genome, postfix: str = ""):
    nn = NeuralNetwork(genome)
    nn.save(f"./topology_genomes/{task_name}/{generation}{postfix}")

def save_fitness_graph(polulation: Population):
    g = np.arange(0, len(population.winners), 1)
    best_fitness_history = [winner.fitness for winner in population.winners]
    average_fitness_history = polulation.average_fitness

    plt.clf()

    plt.plot(g, best_fitness_history, label = "Best") 
    plt.plot(g, average_fitness_history, label = "Average") 
    plt.title(task_name)
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.savefig(f"./graphs/{task_name}")


if __name__ == "__main__":
    population = Population(task)
    task_name = population.task.name
    max_generations = 50
    try:
        for generation in range(max_generations):
            population.evolve()
            print(f"\n---Generation №{generation}--- \
                  \nBest fitness: {population.winners[-1].fitness} \
                  \nGenomes: {len(population.genomes)}; Species: {len(population.species)}")
            if generation % config.save_interval == 0:
                winner = population.winners[-1]
                save_genome(winner)
                nn = NeuralNetwork(winner)
                nn.visualize(show_image=False, save=True, name=f"{task_name}/{generation}", is_solution=False)
            
            if population.solved_id is not None or generation == max_generations - 1:
                winner = population.winners[-1]
                print(f"Solved at {population.solved_id}")
                print(f"Winner fitness {winner.fitness}")
                save_genome(winner, "_solved")
                save_fitness_graph(population)

                nn = NeuralNetwork(winner)
                nn.visualize(show_image=False, save=True, name=f"{task_name}/{generation}", is_solution=True)
                nn.visualize(show_image=False, save=True, name=f"{task_name}/{generation}", is_solution=False)
                population.task.visualize(nn)
                break
    except KeyboardInterrupt:
        save_genome(winner)
        save_fitness_graph(population)
