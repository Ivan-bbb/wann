from feed_forward_nn import NeuralNetwork
from population import Population
from cancer import CancerTask
from diabetes import DiabetesTask
from cart_pole import CartPoleTask
from config import config

task = CartPoleTask

if __name__ == "__main__":
    population = Population(task)
    task_name = population.task.name
    nn = NeuralNetwork.load(f"./topology_genomes/{task_name}/3_solved")

    for weight in config.weights_pool:
        if abs(weight - nn.genome.best_weight) < 0.001:
            print(f"Weight = {weight} (best)")
        else:
            print("Weight =", weight)
        nn.set_weights(weight)
        print("Fitness =", population.task.evaluate(nn))
        nn.visualize(show_image = True, save=False)
        population.task.visualize(nn)
