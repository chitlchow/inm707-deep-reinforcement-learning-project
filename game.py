import pygame
import random

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Set dimensions of screen and grid
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
GRID_SIZE = 20

# Define Snake class


class Snake:
    def __init__(self):
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def move(self):
        x, y = self.positions[0]
        if self.direction == UP:
            y -= GRID_SIZE
        elif self.direction == DOWN:
            y += GRID_SIZE
        elif self.direction == LEFT:
            x -= GRID_SIZE
        elif self.direction == RIGHT:
            x += GRID_SIZE
        self.positions = [(x, y)] + self.positions[:-1]

    def change_direction(self, direction):
        if direction == UP and self.direction != DOWN:
            self.direction = UP
        elif direction == DOWN and self.direction != UP:
            self.direction = DOWN
        elif direction == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        elif direction == RIGHT and self.direction != LEFT:
            self.direction = RIGHT

    def grow(self):
        x, y = self.positions[-1]
        if self.direction == UP:
            y += GRID_SIZE
        elif self.direction == DOWN:
            y -= GRID_SIZE
        elif self.direction == LEFT:
            x += GRID_SIZE
        elif self.direction == RIGHT:
            x -= GRID_SIZE
        self.positions.append((x, y))

    def draw(self, surface):
        for position in self.positions:
            rect = pygame.Rect(position[0], position[1], GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(surface, WHITE, rect)

# Define Food class


class Food:
    def __init__(self):
        self.position = (0, 0)
        self.spawn()

    def spawn(self):
        x = random.randint(0, SCREEN_WIDTH / GRID_SIZE - 1) * GRID_SIZE
        y = random.randint(0, SCREEN_HEIGHT / GRID_SIZE - 1) * GRID_SIZE
        self.position = (x, y)

    def draw(self, surface):
        rect = pygame.Rect(
            self.position[0], self.position[1], GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(surface, RED, rect)


# Initialize Pygame
pygame.init()

# Set up game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Set up clock for controlling frame rate
clock = pygame.time.Clock()

# Define game loop


def game_loop():
    snake = Snake()
    food = Food()
    score = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.change_direction(UP)
                elif event.key == pygame.K_DOWN:
                    snake.change_direction(DOWN)
                elif event.key == pygame.K_LEFT:
                    snake.change_direction(LEFT)
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction(RIGHT)

        # Move snake
        snake.move()

        # Check for collision with food
        if snake.positions[0] == food.position:
            food.spawn()
            snake.grow()
            score +=
