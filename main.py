import pygame
from enum import Enum
import numpy as np
import random
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


class SnakeGame(tf_py_environment.PyEnvironment):
    def __init__(self):
        pygame.init()

    def reset(self):
        print('hi')


# hyperparameters (copied from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)
num_iterations = 20000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


pygame.init()


class ActionSpace(Enum):
    UP = 'up'
    DOWN = 'down'
    RIGHT = 'right'
    LEFT = 'left'


# Setting up variables
width: int = 400
height: int = 400
x1: int = width/2
y1: int = height/2
x1_change: int = 0
y1_change: int = 0
x_mult: int = 0
y_mult: int = 0
score: int = 0
block: int = 10
food_x: int = 100
food_y: int = 100

# Colors
blue = (0, 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)

dis = pygame.display.set_mode((width, height))
pygame.display.update()
pygame.display.set_caption('Snake game by Aumesh')
game_over = False
snake_body = []

step = 0
reward = 0
action_space = [ActionSpace.UP, ActionSpace.DOWN,
                ActionSpace.RIGHT, ActionSpace.LEFT]


def displayGameOver():
    fontTitle = pygame.font.SysFont("arial", 24)
    textTitle = fontTitle.render("Game Over", True, red)
    rectTitle = textTitle.get_rect(center=dis.get_rect().center)
    dis.blit(textTitle, rectTitle)
    pygame.display.update()


clock = pygame.time.Clock()


def playStep():
    next_step: ActionSpace = np.random.choice(action_space, size=1)

    # Random steps
    match next_step:
        case ActionSpace.UP:
            y1_change = -block
            x1_change = 0
        case ActionSpace.DOWN:
            y1_change = block
            x1_change = 0
        case ActionSpace.LEFT:
            x1_change = -block
            y1_change = 0
        case ActionSpace.RIGHT:
            x1_change = block
            y1_change = 0

    # Checking for quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
            return score, reward, game_over

    # Checks out of bounds
    if x1 <= 0 or x1 >= width or y1 <= 0 or y1 >= height:
        displayGameOver()
        reward -= 20
        game_over = True
        return score, reward, game_over

    # Checks for collision with itself
    for body_part in snake_body[:-1]:
        if body_part[0] == x1 and body_part[1] == y1:
            displayGameOver()
            reward -= 20
            game_over = True
            return score, reward, game_over

    x1 += x1_change
    y1 += y1_change
    dis.fill(white)

    # Randomizes food palcement after being eaten
    if (x1 == food_x and y1 == food_y):
        food_x = round(random.randrange(
            0, width - block) / 10.0) * 10.0
        food_y = round(random.randrange(
            0, height - block) / 10.0) * 10.0
        score += 1
        reward += 10

    pygame.draw.rect(dis, red, [food_x, food_y, block, block])

    # Draws snake body
    snake_body.append([x1, y1])
    if (len(snake_body) > score+1):
        del snake_body[0]
    for body_part in snake_body:
        pygame.draw.rect(
            dis, blue, [body_part[0], body_part[1], block, block])

    pygame.display.update()
    clock.tick(20)
    return score, reward, game_over


# pygame.quit()
# quit()
