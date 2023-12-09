import math
import pygame
import random
import numpy as np

# Colors
blue = (0, 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)


class Food:
    food_x: int
    food_y: int

    def __init__(self, width, height):
        self.food_x, self.food_y = width/2 + 10, height/2

    def placeFood(self, width, height):
        food_x = random.randrange(1, (width // 10)) * 10
        food_y = random.randrange(1, (height // 10)) * 10
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
    def __init__(self, mode="train"):
        pygame.init()
        self.width: int = 400
        self.height: int = 400
        self.block = 10
        self.snake = Snake(self.width, self.height)
        self.food = Food(self.width, self.height)
        self.mode = mode

        if (mode != "train"):
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
        obs = [self.snake.x1, self.snake.y1,
               self.food.food_x, self.food.food_y, up, right, down, left]
        obs = np.array(obs, dtype=np.float32)
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
        x_before = self.snake.x1
        y_before = self.snake.y1

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

        if (newDistance < oldDistance):  # if we get closer to food increase reward
            self.reward = 10
        else:
            self.reward = -10

        self.snake.snake_body.append([self.snake.x1, self.snake.y1])
        if (len(self.snake.snake_body) > self.score+3):
            del self.snake.snake_body[0]

        # Randomizes food palcement after being eaten
        if (self.snake.x1 == self.food.food_x and self.snake.y1 == self.food.food_y):
            self.score += 1
            self.reward = 1000
            self.food.food_x, self.food.food_y = self.food.placeFood(
                self.width, self.height)

        # Checks out of bounds
        if self.snake.x1 < 0 or self.snake.x1 >= self.width or self.snake.y1 < 0 or self.snake.y1 >= self.height:
            self.reward = -500
            self.game_over = True

        # Checks for collision with itself
        for body_part in self.snake.snake_body[:-1]:
            if body_part[0] == self.snake.x1 and body_part[1] == self.snake.y1:
                self.reward = -500
                self.game_over = True

        if (self.mode != 'train'):
            self.clock.tick(10)

        return
