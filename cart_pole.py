import numpy as np
import gymnasium as gym

from feed_forward_nn import NeuralNetwork
from base_task import Task


class CartPoleTask(Task):
    def __init__(self):
        self.episodes = 10
        self.env = gym.make("CartPole-v1", max_episode_steps=500)
        self.env.reset()

        self._name = "cartPole"
        self._input_nodes = self.env.observation_space.shape[0]
        self._output_nodes = 2
        self._threshold = 0.9
        print(f"Starting '{self._name}' task with {self._input_nodes} inputs and {self._output_nodes} outputs")
    
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
    
    def normalize_reward(self, reward: float) -> float:
        min_reward = 0
        max_reward = 500
        clamped_reward = np.clip(reward, min_reward, max_reward)
        normalized_reward = (clamped_reward - min_reward) / (max_reward - min_reward)
        return np.clip(normalized_reward, 0, 1)

    def evaluate(self, neural_network: NeuralNetwork) -> float:
        total_reward = 0
        for _ in range(self.episodes):
            state, _ = self.env.reset()
            done = False
            truncate = False
            while not done and not truncate:
                output = neural_network.feed(state)
                action = 0 if output[0] < 0 else 1
                state, reward, done, truncate, _ = self.env.step(action)
                total_reward += reward
        average_reward = self.normalize_reward(total_reward / self.episodes)
        return average_reward

    def solve(self, neural_network: NeuralNetwork) -> bool:
        average_reward = self.evaluate(neural_network)
        # print("avg reward", average_reward)
        return average_reward >= self._threshold

    def visualize(self, neural_network: NeuralNetwork):
        env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=500)
        state, _ = env.reset()
        done = False
        truncate = False
        total_reward = 0
        while not done and not truncate:
            env.render()
            output = neural_network.feed(state)
            action = 0 if output[0] < 0 else 1
            state, reward, done, truncate, info = env.step(action)
            total_reward += reward
        print(f"Total reward: {total_reward}")
        print(f"Normalized reward: {self.normalize_reward(total_reward)}")
        env.close()
