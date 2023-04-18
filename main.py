import random
import numpy as np
import pygame
from src.snake import Snake
from src.food import Food
from src.QLAgent import QLearner
import pandas as pd


# Helper Function
def drawGrid(surface):
    for y in range(0, int(grid_height)):
        for x in range(0, int(grid_width)):
            if (x+y)%2 == 0:
                r = pygame.Rect((x*gridsize, y*gridsize), (gridsize,gridsize))
                pygame.draw.rect(surface,(93,216,228), r)
            else:
                rr = pygame.Rect((x*gridsize, y*gridsize), (gridsize,gridsize))
                pygame.draw.rect(surface, (84,194,205), rr)

def get_state(snake, food):
    # Initialize the array of the states
    states = []
    # 1: States of current direction
    actions = [up, down, right, left]
    for action in actions:
        states.append(int(snake.direction == action))

    # 2: Check food position relative to the snake
    food_direction_states = snake_food_direction(snake, food)

    for dir_state in food_direction_states:
        states.append(int(dir_state))

    # 3: Dangers ahead
    dangers = check_dangers(snake)
    for danger in dangers:
        states.append(int(danger))

    return tuple(states)

def check_dangers(snake):
    x, y = snake.positions[0]
    danger_up = False
    danger_down = False
    danger_right = False
    danger_left = False

    # Check right-hand side
    if x + gridsize >= screen_width or (x + gridsize, y) in snake.positions:
        danger_right = True
    # Check left
    if x - gridsize < 0 or (x - gridsize, y) in snake.positions:
        danger_left = True
    # Check down
    if y + gridsize >= screen_height or (x, y + gridsize) in snake.positions:
        danger_down = True
    # Check up
    if y - gridsize < 0 or (x, y - gridsize) in snake.positions:
        danger_up = True
    dangers = [danger_up, danger_down, danger_right, danger_left]
    return dangers

def snake_food_direction(snake, food):
    head_pos = snake.positions[0]
    food_pos = food.position

    # Compute the difference in coordinates
    delta_x = head_pos[0] - food_pos[0]
    delta_y = head_pos[0] - food_pos[0]
    food_up = False
    food_down = False
    food_right = False
    food_left = False
    if delta_x < 0:
        food_right = True
        food_left = False
    elif delta_x > 0:
        food_right = False
        food_left = True
    if delta_y > 0:
        food_up = False
        food_down = True
    elif delta_y < 0:
        food_up = True
        food_down = False

    return [food_up, food_down, food_right, food_left]


screen_width = 400
screen_height = 400

gridsize = 20
grid_width = screen_width/gridsize
grid_height = screen_height/gridsize

up = (0,-1)
down = (0,1)
left = (-1,0)
right = (1,0)

score = 0
num_episodes = 10000

game_speed = 100000
# Main program for the game
def reset_game(snake, food):
    snake.length = 1
    snake.positions = [(screen_width/2, screen_height/2)]
    food.randomize_position()


def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)
    learner = QLearner(screen_width, screen_height, gridsize)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)

    score = 0
    snake = Snake(screen_width, screen_height)
    food = Food(screen_width, screen_height)

    # Score label on screen
    score_display = pygame.font.SysFont("monospace",16)
    steps_without_food = 0
    episode = 1
    while episode <= num_episodes:
        clock.tick(game_speed)
        # Make move decision
        # Get state and actions
        drawGrid(surface)
        reward = 0
        current_state = get_state(snake, food)
        action = learner.get_action(current_state)
        snake.turn(learner.actions[action])
        crash = snake.move()
        new_state = get_state(snake, food)

        if crash:
            learner.history.append(score)
            # Reset score
            score = 0
            reward = -10
            episode += 1
            new_state = get_state(snake, food)
            learner.update_epsilon()
            reset_game(snake, food)
            if episode%25 ==0:
                # print(learner.Q_tables)

                print("EP: {}, Mean Score: {}, epsilon: {}".format(episode, np.mean(np.array(learner.history)), learner.epsilon))

                learner.history = []


        # Check if the snake eat the food
        if snake.get_head_position() == food.position:
            snake.length += 1
            score += 1
            reward = 1
            # Reset the counter
            steps_without_food = 0
            food.randomize_position()
        else:
            steps_without_food += 1

        if steps_without_food == 1000:
            score = 0
            episode += 1
            learner.update_epsilon()
            if episode%25 ==0:
                print("EP: {}, Mean Score: {}, epsilon: {}".format(episode, np.mean(np.array(learner.history)), learner.epsilon))
            learner.update_epsilon()
            reset_game(snake, food)

        learner.update_Q_valeus(old_state=current_state,
                                new_state=new_state,
                                action=action,
                                reward=reward)
        snake.draw(surface)
        food.draw(surface)

        screen.blit(surface, (0,0))
        # Game running information
        text = score_display.render("Score {0}".format(score), 1, (0,0,0))
        ep = score_display.render("EP: {}".format(episode), 1, (0,0,0))
        screen.blit(text, (5, 10))
        screen.blit(ep, (5, 30))

        pygame.display.update()
main()