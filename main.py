import pygame
from enum import Enum
import numpy as np
import random

pygame.init()


class ActionSpace(Enum):
    UP = 'up'
    DOWN = 'down'
    RIGHT = 'right'
    LEFT = 'left'


# Setting up variables
width: int = 400
height: int = 400
x1 = width/2
y1 = height/2
x1_change = 0
y1_change = 0
x_mult = 0
y_mult = 0
score = 0
block = 10
food_x = 100
food_y = 100

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
action_space = [ActionSpace.UP, ActionSpace.DOWN,
                ActionSpace.RIGHT, ActionSpace.LEFT]


def displayTitle():
    fontTitle = pygame.font.SysFont("arial", 24)
    textTitle = fontTitle.render("Game Over", True, red)
    rectTitle = textTitle.get_rect(center=dis.get_rect().center)
    dis.blit(textTitle, rectTitle)
    pygame.display.update()


clock = pygame.time.Clock()
while not game_over and not step == 100:
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

    # Human moves
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x1_change = -block
                y1_change = 0
            elif event.key == pygame.K_RIGHT:
                x1_change = block
                y1_change = 0
            elif event.key == pygame.K_UP:
                y1_change = -block
                x1_change = 0
            elif event.key == pygame.K_DOWN:
                y1_change = block
                x1_change = 0

    # Process move
    if x1 <= 0 or x1 >= width or y1 <= 0 or y1 >= height:
        displayTitle()
        game_over = True
    else:
        step += 1
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
        pygame.draw.rect(dis, red, [food_x, food_y, block, block])

        # Draws snake body
        snake_body.append([x1, y1])
        if (len(snake_body) > score+1):
            del snake_body[0]

        # Checks for collision with itself
        for body_part in snake_body[:-1]:
            if body_part[0] == x1 and body_part[1] == y1:
                game_over = True
        for body_part in snake_body:
            pygame.draw.rect(
                dis, blue, [body_part[0], body_part[1], block, block])
        pygame.display.update()
        clock.tick(20)


# pygame.quit()
# quit()
