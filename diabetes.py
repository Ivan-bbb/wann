import numpy as np

from feed_forward_nn import NeuralNetwork
from base_task import Task


def init_dataset():
    file_path = "data/diabetes/diabetes.raw"
    data_input = []
    data_output = []
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split(",")
            try:
                data = [float(value) for value in values[:8]]
                labels = [int(value) for value in values[8:]]
            except ValueError:
                continue
            data_input.append(data)
            data_output.append([labels])

    arr = np.array(data_input)
    max_values = np.amax(arr, axis=0)
    for i in range(len(data_input)):
        data_input[i] = data_input[i] / max_values

    return data_input, data_output

def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2).mean()

class DiabetesTask(Task):
    def __init__(self):
        self._threshold = 0.75
        input_data, output_data = init_dataset()
        self.train_data = input_data[:384]
        self.train_labels = output_data[:384]
        self.test_data = input_data[384:]
        self.test_labels = output_data[384:]

        self._name = "diabetes"
        self._input_nodes = 8
        self._output_nodes = 2
        print(f"'{self._name}' task. inputs = {self._input_nodes}; outputs = {self._output_nodes}")
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def input_nodes(self) -> int:
        return self._input_nodes

    @property
    def output_nodes(self) -> int:
        return self._output_nodes
    
    @property
    def threshold(self) -> int:
        return self._threshold

    def evaluate(self, neural_network: NeuralNetwork) -> float:
        total_fitness = 0
        for x_train, y_train in zip(self.train_data, self.train_labels):
            y_pred = neural_network.feed(x_train)
            error = mse(np.array(y_train), np.array(y_pred))
            fitness = 1 / (1 + error)
            total_fitness += fitness

        return total_fitness / len(self.train_data)


    def solve(self, neural_network: NeuralNetwork) -> bool:
        return self.evaluate(neural_network) > self._threshold

    def visualize(self, neural_network: NeuralNetwork):
        correct_train = 0
        wrong_train = 0
        correct_test = 0
        wrong_test = 0

        for x_train, y_train in zip(self.train_data, self.train_labels):
            y_pred = neural_network.feed(x_train)

            if (np.abs(np.array(y_pred) - np.array(y_train)) < 0.5).all():
                correct_train += 1
            else:
                wrong_train += 1

        for x_test, y_test in zip(self.test_data, self.test_labels):
            y_pred = neural_network.feed(x_test)

            if (np.abs(np.array(y_pred) - np.array(y_test)) < 0.5).all():
                correct_test += 1
            else:
                wrong_test += 1

        print(f"Train accuracy = {round(correct_train / len(self.train_data) * 100, 2)}% ({correct_train}/{len(self.train_data)}) \
              \nTest accuracy = {round(correct_test / len(self.test_data) * 100, 2)}% ({correct_test}/{len(self.test_data)})")
