from feed_forward_nn import NeuralNetwork
from population import Population
from wann import Genome
from cancer import CancerTask
from diabetes import DiabetesTask

def save_genome(genome: Genome, postfix: str = ""):
    pass


if __name__ == "__main__":
    population = Population(CancerTask)
    task_name = population.task.name
    max_generations = 300
    for generation in range(max_generations):
        population.evolve()
        print(f"[{generation}] Best fitness: {population.champions[-1].fitness}; Species: {len(population.species)}")
        if generation % 10 == 0:
            save_genome(population.champions[-1])
        
        if population.solved_at is not None or generation == max_generations - 1:
            champion = population.champions[-1]
            print(f"Solved at {population.solved_at}")
            print(f"Champion fitness {champion.fitness}")
            save_genome(champion, "_solved")

            test_nn = NeuralNetwork(champion)
            # test_nn.visualize(show_weights=True, save=True, name=task_name)
            population.task.visualize(test_nn)
            break
