from feed_forward_nn import NeuralNetwork
from population import Population
from cancer import CancerTask
from diabetes import DiabetesTask
from config import config

if __name__ == "__main__":
    population = Population(CancerTask)
    task_name = population.task.name
    nn = NeuralNetwork.load(f"./outputs/{task_name}/0")

    print("Different shared weights test")
    for weight in config.weights_pool:
        if abs(weight - nn.genome.best_weight) < 0.001:
            print("Best weight result")
        nn.set_weights(weight)
        population.task.evaluate(nn)
        # nn.visualize(show_weights=True, save=False)
        population.task.visualize(nn)
