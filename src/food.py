import random
import pygame

class Food:
    def __init__(self, x, y):
        # Initialize the food position and color
        self.x = x
        self.y = y
        self.color = (0, 0, 200)
        self.size = 20

    def move_to_random_position(self, screen):
        screen_info = pygame.display.Info()
        pygame.draw.rect(screen, (255, 255, 255), [self.x, self.y, self.size, self.size])
        self.x = random.randrange(0, screen_info.current_w , 20)
        self.y = random.randrange(0, screen_info.current_h, 20)
        self.show_food(screen)

    def show_food(self, screen):
        pygame.draw.rect(screen, self.color, [self.x, self.y, self.size, self.size])
