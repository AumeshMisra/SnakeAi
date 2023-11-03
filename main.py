import pygame
from enum import Enum


class Axis(Enum):
    X = 'x'
    Y = 'y'


pygame.init()

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

# Colors
blue = (0, 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)

dis = pygame.display.set_mode((width, height))
pygame.display.update()
pygame.display.set_caption('Snake game by Aumesh')
game_over = False
direction = Axis.X


def adjustDirection(directionTowards: Axis, x_mult: int, y_mult: int):
    if (directionTowards == Axis.X):
        if (x_mult < score):
            x_mult += 1
        if (y_mult > 0):
            y_mult -= 1
    elif (directionTowards == Axis.Y):
        if (y_mult < score):
            y_mult += 1
        if (x_mult > 0):
            x_mult -= 1
    return x_mult, y_mult


def displayTitle():
    fontTitle = pygame.font.SysFont("arial", 24)
    textTitle = fontTitle.render("Game Over", True, red)
    rectTitle = textTitle.get_rect(center=dis.get_rect().center)
    dis.blit(textTitle, rectTitle)
    pygame.display.update()


clock = pygame.time.Clock()
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x1_change = -10
                y1_change = 0
                direction = Axis.X
            elif event.key == pygame.K_RIGHT:
                x1_change = 10
                y1_change = 0
                direction = Axis.X
            elif event.key == pygame.K_UP:
                y1_change = -10
                x1_change = 0
                direction = Axis.Y
            elif event.key == pygame.K_DOWN:
                y1_change = 10
                x1_change = 0
                direction = Axis.Y
    x_mult, y_mult = adjustDirection(direction, x_mult, y_mult)
    if x1 <= 0 or x1 >= width:
        displayTitle()
        game_over = True
    elif y1 <= 0 or y1 >= height:
        displayTitle()
        game_over = True
    else:
        x1 += x1_change
        y1 += y1_change
        dis.fill(white)
        if (x1 == 100 and y1 == 100):
            score += 1
            if direction == Axis.X:
                x_mult += 1
            if direction == Axis.Y:
                y_mult += 1
        pygame.draw.rect(dis, red, [100, 100, 10, 10])  # draw food
        pygame.draw.rect(
            dis, blue, [x1, y1, 10 + (10*x_mult), 10 + (10*y_mult)])
        pygame.display.update()
        clock.tick(20)


# pygame.quit()
# quit()
