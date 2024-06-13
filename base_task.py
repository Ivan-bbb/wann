from feed_forward_nn import NeuralNetwork
from abc import ABC, abstractmethod


class Task(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def input_nodes(self):
        pass

    @property
    @abstractmethod
    def output_nodes(self):
        pass

    @property
    @abstractmethod
    def threshold(self):
        pass

    @abstractmethod
    def evaluate(self, neural_network: NeuralNetwork):
        pass

    @abstractmethod
    def solve(self, neural_network: NeuralNetwork):
        pass

    @abstractmethod
    def visualize(self, neural_network: NeuralNetwork):
        pass
