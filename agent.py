from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from snake_env import SnakeGame


class Agent(Env):
    mode: str

    def __init__(self, mode='train'):
        self.snake_game = SnakeGame()
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.mode = mode

    def render(self):
        self.snake_game.view()

    def reset(self):
        del self.snake_game
        self.snake_game = SnakeGame()
        obs = self.snake_game.observe()
        return obs

    def step(self, action: int):
        self.snake_game.action(action, self.mode)
        obs = self.snake_game.observe()
        reward = self.snake_game.evaluate()
        done = self.snake_game.is_done()
        return obs, reward, done, {}
