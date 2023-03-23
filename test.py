import random
import pygame

# Environment Parameters
screen_width = 1280
screen_height = 720
grid_size = 20


# Define a main function
def main():
    # Initialize the pygame module
    pygame.init()
    pygame.display.set_caption("Snake Game")

    screen = pygame.display.set_mode((screen_width, screen_height))
    # Define a variable to control the main loop
    running = True
    # Main loop
    while running:
        # Event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False


class Snake:
    def __init__(self):
        # Starting the snakes at the middle of the screen
        self.position = [screen_width//2, screen_height//2]
        # Initialize a random direction for the snake to start
        self.direction = random.choice(["up","down" ,"left", "right"])
        # Set the color of the snake to Green
        self.color = (0, 200, 0) # Color of the snake

    def move(self):
        x, y = self.position[0]
        if self.direction == "up":
            y -= grid_size
        elif self.direction == 'down':
            y += grid_size
        elif self.direction == 'left':
            x -= grid_size
        else:
            x += grid_size

    def change_direction(self, direction):
        if direction == 'up' and self.direction != 'down':
            self.direction = 'up'
        elif direction == 'down' and self.direction != 'up':
            self.direction = 'down'
        elif direction == 'left' and self.direction != 'right':
            self.direction = 'left'
        elif direction == 'right' and self.direction != 'left':
            self.direction = 'right'





if __name__ == "__main__":
    main()