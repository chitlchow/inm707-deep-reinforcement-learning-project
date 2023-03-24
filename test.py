import random
import pygame

# Environment Parameters
screen_width = 1280
screen_height = 720
grid_size = 20
black = (0, 0, 0)

# Game clock
clock = pygame.time.Clock()

class Snake:
    def __init__(self):
        # Starting the snakes at the middle of the screen
        self.x = screen_width//2
        self.y = screen_height//2
        # Initialize a random direction for the snake to start
        self.direction = random.choice(["up","down" ,"left", "right"])
        # Set the color of the snake to Green
        self.color = (0, 200, 0) # Color of the snake

    def move(self):
        x, y = self.x, self.y
        if self.direction == "up":
            y -= grid_size
        elif self.direction == 'down':
            y += grid_size
        elif self.direction == 'left':
            x -= grid_size
        else:
            x += grid_size
        if x > 1280:
            x = 0
        if x < 0:
            x = 1280
        if y > 720:
            y = 0
        if y < 0:
            y = 720
        return x, y

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
        pygame.draw.rect(screen, self.color, [self.x, self.y, grid_size, grid_size])


class Food:
    def __init__(self):
        self.x = random.randrange(0, 1280, 20)
        self.y = random.randrange(0, 720, 20)
        self.color = (0, 0, 200)

    def move_to_random_position(self):
        self.x = random.randint(0, 1280)
        self.y = random.randint(0, 720)

    def show_food(self, screen):
        pygame.draw.rect(screen, self.color, [self.x, self.y, grid_size, grid_size])


# Define a main function
def main():
    # Initialize the pygame module
    pygame.init()
    pygame.display.set_caption("Snake Game")
    snake = Snake()
    food = Food()
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
        snake.show_snake(screen)
        food.show_food(screen)
        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[pygame.K_UP]:
            direction = 'up'
        elif keys_pressed[pygame.K_DOWN]:
            direction = 'down'
        elif keys_pressed[pygame.K_RIGHT]:
            direction = 'right'
        elif keys_pressed[pygame.K_LEFT]:
            direction = 'left'
        else:
            direction = snake.direction
        snake.change_direction(direction)
        pygame.draw.rect(screen, black, [snake.x, snake.y, grid_size, grid_size])
        snake.x, snake.y = snake.move()
        snake.show_snake(screen)
        pygame.display.update()
        print(snake.x, snake.y)
        print(food.x, food.y)
        clock.tick(10)






if __name__ == "__main__":
    main()