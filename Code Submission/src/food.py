import random
import pygame

gridsize = 20
class Food():
    def __init__(self, screen_width, screen_height):
        self.position = (0,0)
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.color = (223, 163, 49)
        self.randomize_position()

    def randomize_position(self):
        grid_width = self.screen_width / gridsize
        grid_height = self.screen_height / gridsize
        self.position = (random.randint(0, grid_width-1)*gridsize, random.randint(0, grid_height-1)*gridsize)

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (gridsize, gridsize))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, (93, 216, 228), r, 1)