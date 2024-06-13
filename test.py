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
    nn = NeuralNetwork.load(f"./topology_genomes/{task_name}/8_solved")

    # test_only_on_best_weight = False

    for weight in config.weights_pool:
        # if weight == nn.genome.best_weight and test_only_on_best_weight:
        #     print(f"Weight = {weight} (best)")
        #     nn.set_weights(weight)
        #     population.task.evaluate(nn)
        #     print("Fitness =", population.task.evaluate(nn))
        #     nn.visualize(show_image = True, save=False)
        #     population.task.visualize(nn)
        # elif not test_only_on_best_weight:
        if abs(weight - nn.genome.best_weight) < 0.001:
            print(f"Weight = {weight} (best)")
        else:
            print("Weight =", weight)
        nn.set_weights(weight)
        # population.task.evaluate(nn)
        print("Fitness =", population.task.evaluate(nn))
        nn.visualize(show_image = True, save=False)
        population.task.visualize(nn)
