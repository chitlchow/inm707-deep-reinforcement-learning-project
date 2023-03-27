import random
import pygame
from src.snake import Snake
from src.food import Food


# Environment Parameters
screen_width = 500
screen_height = 500
grid_size = 20
black = (0, 0, 0)
white = (255, 255, 255)
# Game clock
clock = pygame.time.Clock()


# Define a main function
def main():
    # Initialize the pygame module
    pygame.init()
    pygame.display.set_caption("Snake Game")
    snake = Snake()
    food = Food(x=random.randrange(0, screen_width, 20), y=random.randrange(0, screen_height, 20))
    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill(white)
    score = 0
    # Define a variable to control the main loop
    game_over = False

    # Main loop
    while not game_over:
        snake.show_snake(screen)
        food.show_food(screen)

        # Event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                game_over = False

        # Check if the snake eat the food - means they have the same coordinate
        if (snake.x, snake.y) == (food.x, food.y):
            # Add score
            score += 1
            # Move the food object to a different position
            food.move_to_random_position(screen)
            # Print out the score that the snake got
            print("Score: {}".format(score))

        # Get key pressed
        keys_pressed = pygame.key.get_pressed()

        # Move the snake according the directional key pressed
        # The pygame.KEY_Name returns the ASCII number of the key
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

        # If the snake hit the boarder, terminate game and break the loop
        if snake.hit_boarder():
            pygame.quit()
            break
        snake.change_direction(direction)
        pygame.draw.rect(screen, white, [snake.x, snake.y, grid_size, grid_size])
        snake.x, snake.y = snake.move()
        snake.show_snake(screen)
        pygame.display.update()
        clock.tick(10)

if __name__ == "__main__":
    main()