# from keras import __version__  # nopep8
# import tensorflow  # nopep8
# tensorflow.keras.__version__ = __version__  # nopep8

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import pygame
import numpy as np
import random
from gym import Env
from gym.spaces import Discrete, Box
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam


# Colors
blue = (0, 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)


class Food:
    food_x: int
    food_y: int

    def __init__(self, width, height, block):
        self.food_x, self.food_y = self.placeFood(width, height, block)

    def placeFood(self, width, height, block):
        food_x = round(random.randrange(
            0, width - block) / 10.0) * 10.0
        food_y = round(random.randrange(
            0, height - block) / 10.0) * 10.0
        return (food_x, food_y)


class Snake:
    x1: int
    y1: int
    snake_body: list

    def __init__(self, width, height):
        self.x1 = width/2
        self.y1 = height/2
        self.snake_body = []
        self.direction = 0  # 0 is up, 1 is right, 2 is down, 3 is left


class SnakeGame:
    def __init__(self):
        pygame.init()
        self.width: int = 400
        self.height: int = 400
        self.block = 10
        self.snake = Snake(self.width, self.height)
        self.food = Food(self.width, self.height, self.block)

        self.clock = pygame.time.Clock()
        self.dis = None

        self.score = 0
        self.reward = 0
        self.game_over = False
        self.dis = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake game by Aumesh')

    def view(self):
        # Checking for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                return

        # Draws food
        self.dis.fill(white)
        pygame.draw.rect(
            self.dis, red, [self.food.food_x, self.food.food_y, self.block, self.block])

        # Draws snake body
        self.snake.snake_body.append([self.snake.x1, self.snake.y1])
        if (len(self.snake.snake_body) > self.score+1):
            del self.snake.snake_body[0]
        for body_part in self.snake.snake_body:
            pygame.draw.rect(
                self.dis, blue, [body_part[0], body_part[1], self.block, self.block])
        pygame.display.update()
        return

    def evaluate(self):
        return self.reward

    def observe(self):
        delta_food_x = self.food.food_x - self.snake.x1
        delta_food_y = self.food.food_y - self.snake.y1
        delta_left_wall = self.snake.x1
        delta_right_wall = self.width - self.snake.x1
        delta_bottom = self.snake.y1
        delta_top = self.height - self.snake.y1
        up: int = self.snake.direction == 0
        right: int = self.snake.direction == 1
        down: int = self.snake.direction == 2
        left: int = self.snake.direction == 3
        obs = [self.snake.x1, self.snake.y1, delta_food_x, delta_food_y, self.score,
               delta_left_wall, delta_right_wall, delta_bottom, delta_top, up, right, down, left]
        obs = np.array(obs)
        return obs

    def is_done(self) -> bool:
        # Checks out of bounds
        if self.snake.x1 <= 0 or self.snake.x1 >= self.width or self.snake.y1 <= 0 or self.snake.y1 >= self.height:
            self.reward -= 100
            self.game_over = True
            return self.game_over

        # Checks for collision with itself
        for body_part in self.snake.snake_body[:-1]:
            if body_part[0] == self.snake.x1 and body_part[1] == self.snake.y1:
                self.reward -= 100
                self.game_over = True
                return self.game_over

        return self.game_over

    def action(self, action):
        match action:
            case 0:
                self.snake.y1 += -self.block
                self.snake.direction = 2
            case 1:
                self.snake.y1 += self.block
                self.snake.direction = 0
            case 2:
                self.snake.x1 += -self.block
                self.snake.direction = 3
            case 3:
                self.snake.x1 += self.block
                self.snake.direction = 1

        # Randomizes food palcement after being eaten
        if (self.snake.x1 == self.food.food_x and self.snake.y1 == self.food.food_y):
            self.food.food_x, self.food.food_y = self.food.placeFood(
                self.width, self.height, self.block)
            self.score += 1
            self.reward += 10

        self.clock.tick(20)
        return


class Agent(Env):
    def __init__(self):
        self.snake_game = SnakeGame()
        self.action_space = Discrete(4)
        # we need to define observation space to be the list of states around our snake, food, and walls
        # the observation space is what we observe with respect to the game
        self.observation_space = Box(low=-400, high=400, shape=(13,))

    def render(self):
        self.snake_game.view()

    def reset(self):
        del self.snake_game
        self.snake_game = SnakeGame()
        obs = self.snake_game.observe()
        return obs

    def step(self, action: int):
        self.snake_game.action(action)
        obs = self.snake_game.observe()
        reward = self.snake_game.evaluate()
        done = self.snake_game.is_done()
        return obs, reward, done, {}


env = Agent()

# testing episodes
# episodes = 10
# for episode in range(1, episodes):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))


def build_model(actions):
    model = Sequential()
    model.add(InputLayer(input_shape=(1, 13), name='Input Layer'))
    model.add(Dense(24, activation='relu',
              input_shape=(1, 13), name='Hidden Layer 1'))
    model.add(Dense(actions, activation='relu', name='Hidden Layer 2'))
    return model


states = env.observation_space.shape
actions = env.action_space.n
print(actions)
model = build_model(actions)


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=10000, visualize=True, verbose=1)
