import numpy as np
import gymnasium as gym

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
print("The Maximum of each index list is:", max_values)

print(data_input[0] / max_values)

for i in range(len(data_input)):
    data_input[i] = data_input[i] / max_values

print(data_input[:2])