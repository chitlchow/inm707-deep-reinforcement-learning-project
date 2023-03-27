import random
import pygame


class Snake:
    def __init__(self):
        # Starting the snakes at the middle of the screen
        self.x = 240
        self.y = 240
        # Initialize a random direction for the snake to start
        self.direction = random.choice(["up","down" ,"left", "right"])
        # Set the color of the snake to Green
        self.color = (0, 200, 0) # Color of the snake
        self.size = 20
        self.screen_info = pygame.display.Info()

    # This function is purposed to move the snake along a single direction
    def move(self):
        # Get the current x and y position
        x, y = self.x, self.y
        # Change the position based on the direction it is moving
        if self.direction == "up":
            y -= self.size
        elif self.direction == 'down':
            y += self.size
        elif self.direction == 'left':
            x -= self.size
        else:
            x += self.size
        # Return the x and y coordinates
        return x, y

    def hit_boarder(self):
        # Get the screen information
        if self.x > self.screen_info.current_w or self.x < 0 \
            or self.y > self.screen_info.current_h or self.y < 0:
            return True
        return False

    def change_direction(self, direction):
        if direction == 'up' and self.direction != 'down':
            self.direction = 'up'
        elif direction == 'down' and self.direction != 'up':
            self.direction = 'down'
        elif direction == 'left' and self.direction != 'right':
            self.direction = 'left'
        elif direction == 'right' and self.direction != 'left':
            self.direction = 'right'

    def show_snake(self, screen):
        pygame.draw.rect(screen, self.color, [self.x, self.y, self.size, self.size])

    # The sensor methods returns the state that it currently in:
    # def state(self):
    #     right
    #     left