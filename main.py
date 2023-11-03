import pygame
from enum import Enum

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
block = 10

# Colors
blue = (0, 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)

dis = pygame.display.set_mode((width, height))
pygame.display.update()
pygame.display.set_caption('Snake game by Aumesh')
game_over = False
direction = Axis.X
snake_body = []


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
    if x1 <= 0 or x1 >= width or y1 <= 0 or y1 >= height:
        displayTitle()
        game_over = True
    else:
        x1 += x1_change
        y1 += y1_change
        dis.fill(white)
        if (x1 == 100 and y1 == 100):
            score += 1
        pygame.draw.rect(dis, red, [100, 100, block, block])  # draw food
        snake_body.append([x1, y1])
        if (len(snake_body) > score+1):
            del snake_body[0]
        for x in snake_body:
            pygame.draw.rect(
                dis, blue, [x[0], x[1], block, block])
        pygame.display.update()
        clock.tick(20)


# pygame.quit()
# quit()
