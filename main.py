import pygame
from enum import Enum
import numpy as np
import random
import tensorflow as tf

# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.drivers import py_driver
# from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
# from tf_agents.eval import metric_utils
# from tf_agents.metrics import tf_metrics
# from tf_agents.networks import sequential
# from tf_agents.policies import py_tf_eager_policy
# from tf_agents.policies import random_tf_policy
# from tf_agents.replay_buffers import reverb_replay_buffer
# from tf_agents.replay_buffers import reverb_utils
# from tf_agents.trajectories import trajectory
# from tf_agents.specs import tensor_spec
# from tf_agents.utils import common

# Colors
blue = (0, 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)

# hyperparameters (copied from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)
num_iterations = 20000  # @param {type:"integer"}

initial_collect_step_counts = 100  # @param {type:"integer"}
collect_step_counts_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


class ActionSpace(Enum):
    UP = 'up'
    DOWN = 'down'
    RIGHT = 'right'
    LEFT = 'left'


class SnakeGame(tf_py_environment.TFPyEnvironment):
    def __init__(self):
        pygame.init()
        self.width: int = 400
        self.height: int = 400
        self.x1 = self.width/2
        self.y1 = self.height/2
        self.clock = pygame.time.Clock()
        self.dis = None
        self.food_x = 0
        self.food_y = 0
        self.block = 10
        self.snake_body = []
        self.score = 0
        self.step_count = 0
        self.reward = 0
        self.game_over = False
        self.action_space = [ActionSpace.UP, ActionSpace.DOWN,
                             ActionSpace.RIGHT, ActionSpace.LEFT]

    def reset(self):
        self.dis = pygame.display.set_mode((self.width, self.height))
        pygame.display.update()
        pygame.display.set_caption('Snake game by Aumesh')

    def displayGameOver(self):
        fontTitle = pygame.font.SysFont("arial", 24)
        textTitle = fontTitle.render("Game Over", True, red)
        rectTitle = textTitle.get_rect(center=self.dis.get_rect().center)
        self.dis.blit(textTitle, rectTitle)
        pygame.display.update()

    def placeFood(self):
        self.food_x = round(random.randrange(
            0, self.width - self.block) / 10.0) * 10.0
        self.food_y = round(random.randrange(
            0, self.height - self.block) / 10.0) * 10.0

    def playstep_count(self):
        next_step_count: ActionSpace = np.random.choice(
            self.action_space, size=1)

        # Random step_counts
        match next_step_count:
            case ActionSpace.UP:
                self.y1 += -self.block
            case ActionSpace.DOWN:
                self.y1 += self.block
            case ActionSpace.LEFT:
                self.x1 += -self.block
            case ActionSpace.RIGHT:
                self.x1 += self.block

        # Checking for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                return self.score, self.reward, self.game_over

        # Checks out of bounds
        if self.x1 <= 0 or self.x1 >= self.width or self.y1 <= 0 or self.y1 >= self.height:
            self.displayGameOver()
            self.reward -= 20
            self.game_over = True
            return self.score, self.reward, self.game_over

        # Checks for collision with itself
        for body_part in self.snake_body[:-1]:
            if body_part[0] == self.x1 and body_part[1] == self.y1:
                self.displayGameOver()
                self.reward -= 20
                self.game_over = True
                return self.score, self.reward, self.game_over

        self.dis.fill(white)

        # Randomizes food palcement after being eaten
        if (self.x1 == self.food_x and self.y1 == self.food_y):
            self.food_x = round(random.randrange(
                0, self.width - self.block) / 10.0) * 10.0
            self.food_y = round(random.randrange(
                0, self.height - self.block) / 10.0) * 10.0
            self.score += 1
            self.reward += 10

        pygame.draw.rect(
            self.dis, red, [self.food_x, self.food_y, self.block, self.block])

        # Draws snake body
        self.snake_body.append([self.x1, self.y1])
        if (len(self.snake_body) > self.score+1):
            del self.snake_body[0]
        for body_part in self.snake_body:
            pygame.draw.rect(
                self.dis, blue, [body_part[0], body_part[1], self.block, self.block])

        pygame.display.update()
        self.clock.tick(20)
        return self.score, self.reward, self.game_over


snake_game = SnakeGame()
snake_game.reset()
snake_game.placeFood()

for i in range(0, 100):
    snake_game.playstep_count()
