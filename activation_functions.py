from enum import Enum, member
import numpy as np


def sigmoid(x):
    return (np.tanh(x / 2.0) + 1.0) / 2.0

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def inverse(x):
    return -x

def abs(x):
    return np.abs(x)

def square(x):
    return x * x

def unsigned_step(x):
    return 1.0 * (x > 0.0)

def sin(x):
    return np.sin(np.pi * x)

def cos(x):
    return np.cos(np.pi * x)

def gaussian(x):
    return np.exp(-np.multiply(x, x) / 2.0)


class ActivationFunction(Enum):
    Sigmoid = member(sigmoid)
    Tanh = member(tanh)
    ReLU = member(relu)
    Linear = member(linear)
    Inverse = member(inverse)
    Abs = member(abs)
    Square = member(square)
    Unsigned_step = member(unsigned_step)
    Sin = member(sin)
    Cos = member(cos)
    Gaussian = member(gaussian)
    