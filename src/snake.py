import random
import pygame
import sys

up = (0,-1)
down = (0,1)
left = (-1,0)
right = (1,0)
gridsize = 20

class Snake():
    def __init__(self, screen_width, screen_height):
        self.length = 1
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.positions = [((screen_width/2), (screen_height/2))]
        self.direction = random.choice([up, down, left, right])
        self.color = (17, 24, 47)
        # Special thanks to YouTubers Mini - Cafetos and Knivens Beast for raising this issue!
        # Code adjustment courtesy of YouTuber Elija de Hoog

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0]*-1, point[1]*-1) == self.direction:
            return
        else:
            self.direction = point

    # Return True if crash
    def move(self):
        cur = self.get_head_position()
        x,y = self.direction
        new = (
                (cur[0] + (x * gridsize)),
                (cur[1] + (y * gridsize))  # % pygame.display.Info().current_h
        )
        # Reset if the snake eat itself
        if len(self.positions) > 2 and new in self.positions[2:]:
            return True

        elif new[0] > self.screen_width \
                or new[1] > self.screen_height \
                or new[0] < 0\
                or new[1] < 0:
            return True
        else:
            self.positions.insert(0,new)
            if len(self.positions) > self.length:
                self.positions.pop()

        return False


    def draw(self,surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (gridsize,gridsize))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, (93,216, 228), r, 1)


