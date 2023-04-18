import random
import numpy as np
import pygame
from src.snake import Snake
from src.food import Food
from src.QLAgent import QLearner
import pandas as pd
import os

# Helper Function
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

def coords_to_index(x, y):
    r = int(y//10)
    c = int(x//10)
    return (r,c)

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

game_board = np.zeros((screen_width // gridsize, screen_height //gridsize))

score = 0
num_episodes = 10000

game_speed = 100000
# Main program for the game
def main():
    # Initialize Snake and Food
    snake = Snake(screen_width, screen_height)
    food = Food(screen_width, screen_height)
    snake_r, snake_c = snake.positions[0]
    food_r, food_c = food.position
    # Initialize on the game board
    game_board[snake_r][snake_c] = 1
    game_board[food_r][food_c] = 2
    os.system('clear')

    print("Snake Game")
    print(game_board)





