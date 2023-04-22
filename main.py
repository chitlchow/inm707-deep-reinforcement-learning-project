import numpy as np
import pygame
from src.snake import Snake
from src.food import Food
from src.QLAgent import QLearner
import pandas as pd
import pickle
import time


# Helper Function
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
    # head position
    x, y = snake.positions[0]
    danger_ahead = False
    danger_right = False
    danger_left = False

    actions = [up, right, down, left]
    # Check danger ahead
    front_pos = (x + snake.direction[0]*gridsize, y + snake.direction[1]*gridsize)
    # If it crash it selfs
    if  front_pos in snake.positions \
        or front_pos[0] >= screen_width \
        or front_pos[1] >= screen_height \
        or front_pos[0] < 0 \
        or front_pos[1] < 0:
        danger_ahead = True

    #  Check right
    current_dir_index  = actions.index(snake.direction)
    right_step = actions[(current_dir_index + 1) % 4]
    right_pos = (x + right_step[0]*gridsize, y + right_step[1]*gridsize)
    if right_pos in snake.positions \
        or right_pos[0] >= screen_width \
        or right_pos[1] >= screen_height \
        or right_pos[0] < 0 \
        or right_pos[1] < 0:
        danger_right = True

    # Check left
    left_step = actions[(current_dir_index - 1) % 4]
    left_pos = (x + left_step[0]*gridsize, y + left_step[1]*gridsize)

    if left_pos in snake.positions \
        or left_pos[0] >= screen_width \
        or left_pos[1] >= screen_height \
        or left_pos[0] < 0 \
        or left_pos[1] < 0:
        danger_left = True

    dangers = [danger_ahead, danger_right, danger_left]
    return dangers

def snake_food_direction(snake, food):
    head_pos = snake.positions[0]
    food_pos = food.position

    # Compute the difference in coordinates
    delta_x = head_pos[0] - food_pos[0]
    delta_y = head_pos[0] - food_pos[0]

    # Prepare variables
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

def reset_game(snake, food):
    snake.length = 1
    snake.positions = [(screen_width/2, screen_height/2)]
    food.randomize_position()

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
game_speed = 10000
# Main program for the game

alpha = 0.01
gamma = 0.95
epsilon_discount = 0.9992
time_steps = []
def game_loop(alpha, gamma, epsilon_discount):

    clock = pygame.time.Clock()
    # screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)

    learner = QLearner(screen_width, screen_height, gridsize, alpha, gamma, epsilon_discount)
    # surface = pygame.Surface(screen.get_size())
    # surface = surface.convert()
    # drawGrid(surface)
    high = 0
    score = 0
    snake = Snake(screen_width, screen_height)
    food = Food(screen_width, screen_height)
    training_history = []
    # Score label on screen
    # score_display = pygame.font.SysFont("monospace",16)
    steps_without_food = 0
    episode = 1

    for episode in range(1, num_episodes + 1):
        # print(episode)
        score = 0
        steps_without_food = 0
        start = time.time()
        crash = False
        learner.episode_history = []
        learner.reward_history = []
        learner.clear_history()
        food.randomize_position()
        while not crash or steps_without_food == 1000:
            clock.tick(game_speed)
            reward = 0
            current_state = get_state(snake, food)
            action = learner.get_action(current_state)
            snake.turn(learner.actions[action])
            crash = snake.move()
            new_state = get_state(snake, food)

            if crash or steps_without_food == 1000:
                reward = -10

            if snake.get_head_position() == food.position:
                reward = 10
                score += 1
                if score > high:
                    high = score
                snake.length += 1
                steps_without_food = 0
                food.randomize_position()
            else:
                steps_without_food += 1
            # Update Q-tables
            learner.update_Q_valeus(current_state, new_state, action, reward)
        end = time.time()
        ep_time = end - start()
        learner.history.append(score)
        learner.update_epsilon()
        reset_game(snake, food)
        if episode % 25 ==0:
            print("EP: {}, Mean Score: {}, epsilon: {}, Highest: {}".format(
                episode,
                np.mean(np.array(learner.history)),
                learner.epsilon,
                high
            ))
            with open("training_history/episode-{}.pickle".format(episode), 'wb') as file:
                pickle.dump(learner.Q_tables, file)
            learner.history = []


            # Reset score
        training_history.append((episode, score, learner.epsilon))

    training_history = pd.DataFrame(training_history, columns=['Episodes', 'Score', 'Epsilon'])
    training_history.to_csv('result-dataset/training_history-({}, {}, {}).csv'.format(alpha, gamma, epsilon_discount))

game_loop(alpha, gamma, epsilon_discount)