# from keras import __version__  # nopep8
# import tensorflow  # nopep8
# tensorflow.keras.__version__ = __version__  # nopep8

import math
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import pygame
import numpy as np
import random
from gym import Env
from gym.spaces import Discrete, Box
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten
from keras.optimizers.legacy import Adam


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
        self.snake_body = [[
            self.x1-20, self.y1], [self.x1-10, self.y1], [self.x1, self.y1]]
        self.direction = 1  # 0 is up, 1 is right, 2 is down, 3 is left


class SnakeGame:
    def __init__(self):
        pygame.init()
        self.width: int = 200
        self.height: int = 200
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
        for body_part in self.snake.snake_body:
            pygame.draw.rect(
                self.dis, blue, [body_part[0], body_part[1], self.block, self.block])
        pygame.display.update()
        return

    def evaluate(self):
        return self.reward

    def observe(self):
        delta_food_x = self.food.food_x - self.snake.x1
        delta_food_y = self.snake.y1 - self.food.food_y
        food_above = int(delta_food_y > 0)
        food_below = int(delta_food_y < 0)
        food_right = int(delta_food_x > 0)
        food_left = int(delta_food_x < 0)
        delta_left_wall = self.snake.x1
        delta_right_wall = self.width - self.snake.x1
        delta_bottom = self.height - self.snake.y1
        delta_top = self.snake.y1

        # checking body position:
        # body close
        body_up = []
        body_right = []
        body_down = []
        body_left = []
        if len(self.snake.snake_body) > 2:
            for [x1, y1] in self.snake.snake_body[:-3]:
                if math.sqrt((x1-self.snake.x1)**2 + (y1-self.snake.y1)**2) == 10:
                    if y1 > self.snake.y1:
                        body_down.append(1)
                    elif y1 < self.snake.y1:
                        body_up.append(1)
                    if x1 < self.snake.x1:
                        body_left.append(1)
                    elif x1 > self.snake.x1:
                        body_right.append(1)

        if len(body_up) > 0:
            body_up = 1
        else:
            body_up = 0
        if len(body_right) > 0:
            body_right = 1
        else:
            body_right = 0
        if len(body_down) > 0:
            body_down = 1
        else:
            body_down = 0
        if len(body_left) > 0:
            body_left = 1
        else:
            body_left = 0

        wall_left, wall_right, wall_up, wall_down = 0, 0, 0, 0

        if delta_left_wall == -10:
            wall_left = 1
        if delta_right_wall == 10:
            wall_right = 1
        if delta_bottom == 10:
            wall_down = 1
        if delta_top == 10:
            wall_up = 1

        obstacle_left = int(wall_left or body_left)
        obstacle_right = int(wall_right or body_right)
        obstacle_up = int(wall_up or body_up)
        obstacle_down = int(wall_down or body_down)

        up: int = int(self.snake.direction == 0)
        right: int = int(self.snake.direction == 1)
        down: int = int(self.snake.direction == 2)
        left: int = int(self.snake.direction == 3)
        obs = [self.snake.x1, self.snake.y1, food_above, food_below, food_left, food_right, self.score,
               delta_left_wall, delta_right_wall, obstacle_up, obstacle_right, obstacle_down, obstacle_left, up, right, down, left,
               self.food.food_x, self.food.food_y]
        obs = np.array(obs)
        return obs

    def is_done(self) -> bool:
        return self.game_over

    def makeMove(self, action):
        match action:
            case 0:
                if (self.snake.direction != 2):
                    self.snake.direction = 0
            case 1:
                if (self.snake.direction != 3):
                    self.snake.direction = 1
            case 2:
                if (self.snake.direction != 0):
                    self.snake.direction = 2
            case 3:
                if (self.snake.direction != 1):
                    self.snake.direction = 3

    def action(self, action):
        y_before = self.snake.x1
        x_before = self.snake.y1
        self.makeMove(action)
        match self.snake.direction:
            case 0:
                self.snake.y1 += -self.block
                self.snake.direction = 0
            case 1:
                self.snake.x1 += self.block
                self.snake.direction = 1
            case 2:
                self.snake.y1 += self.block
                self.snake.direction = 2
            case 3:
                self.snake.x1 += -self.block
                self.snake.direction = 3

        oldDistance = math.sqrt((x_before - self.food.food_x)
                                ** 2 + (y_before - self.food.food_y)**2)

        newDistance = math.sqrt((self.snake.x1 - self.food.food_x)
                                ** 2 + (self.snake.y1 - self.food.food_y)**2)

        if (oldDistance > newDistance):
            self.reward += 1
        else:
            self.reward -= 1

        self.snake.snake_body.append([self.snake.x1, self.snake.y1])
        if (len(self.snake.snake_body) > self.score+3):
            del self.snake.snake_body[0]

        # Randomizes food palcement after being eaten
        if (self.snake.x1 == self.food.food_x and self.snake.y1 == self.food.food_y):
            self.food.food_x, self.food.food_y = self.food.placeFood(
                self.width, self.height, self.block)
            self.score += 1
            self.reward += 10

        # Checks out of bounds
        if self.snake.x1 <= 0 or self.snake.x1 >= self.width or self.snake.y1 <= 0 or self.snake.y1 >= self.height:
            self.reward -= 100
            self.game_over = True

        # Checks for collision with itself
        for body_part in self.snake.snake_body[:-1]:
            if body_part[0] == self.snake.x1 and body_part[1] == self.snake.y1:
                self.reward -= 100
                self.game_over = True

        # self.clock.tick(100)
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
#         # print(env.snake_game.snake.direction)
#         # print(n_state)
#     print('Episode:{} Score:{}'.format(episode, score))


def build_model(actions):
    model = Sequential()
    model.add(Dense(128, activation='relu',
              input_shape=(1, 19), name='hl1'))
    model.add(Dense(128, activation='relu', name='hl2'))
    model.add(Dense(128, activation='relu', name='hl3'))
    model.add(Flatten())
    model.add(Dense(actions, activation='softmax', name='hl4'))
    return model


states = env.observation_space.shape
actions = env.action_space.n
model = build_model(actions)
# print(model.summary())


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=2500, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=100, gamma=0.95, batch_size=500)
    # dqn.model.epsilon =
    return dqn


json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
dqn = build_agent(loaded_model, actions)
dqn.model.load_weights('./weights/weights_3.h5')
# # dqn.model.load('./weights/weights.keras')
# # print(dqn)
dqn.compile(Adam(learning_rate=0.00025), metrics=['mse'])
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)

model_json = dqn.model.to_json()
with open("./model/model.json", "w") as json_file:
    json_file.write(model_json)
dqn.model.save_weights('./weights/weights_4.h5', overwrite=True)
