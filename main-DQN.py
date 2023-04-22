import numpy as np
import pygame
from src.snake import Snake
from src.food import Food
from src.DQN import DQN_Agent
import pandas as pd
import pickle
import time

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
    front_pos = (x + snake.direction[0], y + snake.direction[1])
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
    right_pos = (x + right_step[0], y + right_step[1])
    if right_pos in snake.positions \
        or right_pos[0] >= screen_width \
        or right_pos[1] >= screen_height \
        or right_pos[0] < 0 \
        or right_pos[1] < 0:
        danger_right = True

    # Check left
    left_step = actions[(current_dir_index - 1) % 4]
    left_pos = (x + left_step[0], y + left_step[1])

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
num_episodes = 30000
game_speed = 1000

alpha = 0.01
gamma = 0.95
epsilon_discount = 0.9992

graphics = False

# Main program for the game

def game_loop(alpha, gamma, epsilon_discount):
    learner = DQN_Agent(learning_rate=alpha, gamma=gamma, epsilon_decay=epsilon_discount)
    clock = pygame.time.Clock()
    snake = Snake(screen_width, screen_height)
    food = Food(screen_width, screen_height)
    if graphics:
        screen = pygame.display.set_mode((screen_width, screen_height))
        surface = pygame.Surface(screen.get_size())
        surface.convert()
        drawGrid(surface)

    training_histories = []
    start = time.time()
    for episode in range(1, num_episodes + 1):

        crash = False
        score = 0
        steps_without_food = 0
        start = time.time()

        learner.episode_history = []
        learner.reward_history = []
        learner.clear_memory()


        while not crash or steps_without_food == 1000:
            reward = 0
            clock.tick(game_speed)



            # Agent making moves
            current_state = get_state(snake, food)
            action = learner.get_action(current_state)
            snake.turn(learner.actions[action])
            crash = snake.move()
            new_state = get_state(snake, food)
            # Check if the snake eat the food
            if snake.get_head_position() == food.position:
                snake.length += 1
                score += 1
                reward = 10
                # Reset the counter
                steps_without_food = 0

                # new_state = get_state(snake, food)
                food.randomize_position()
            else:
                steps_without_food += 1

            # Case where episodes is going to be terminated
            if crash or steps_without_food == 1000:
                # Break the loop if crashing or timeout
                reward = -10


            # Update the Q-values

            learner.reward_history.append(reward)

            # Memorize step
            learner.memorize(current_state, reward=reward, action=action, new_state=new_state)
            # learner.train_step(current_state, action, reward, new_state)
            learner.train_short_memories()
            if graphics:
                drawGrid(surface)
                snake.draw(surface)
                food.draw(surface)
                screen.blit(surface, (0,0))
                pygame.display.update()

        learner.train_long_memories()
        learner.clear_episode_memories()
        end = time.time()
        ep_time = end - start
        training_histories.append((episode, score, learner.epsilon, ep_time))

        # Reset the environment
        learner.score_history.append(score)
        learner.update_epsilon()
        reset_game(snake, food)
        # print("EP: {}, Score: {}".format(episode, score))
        if episode % 25 == 0:
            print("EP: {}, Mean Score: {:.2f}, epsilon: {:.4f}, episode time: {:.6f}"\
                  .format(episode,np.mean(np.array(learner.score_history)),learner.epsilon, ep_time))
        # with open("training_history/episode-{}.pickle".format(episode), 'wb') as file:
                # pickle.dump(learner.Q_tables, file)
                # pass
            # Clear the history of last 25 episodes
            learner.score_history = []

    training_history = pd.DataFrame(training_histories, columns=['Episodes', 'Score', 'Epsilon', 'Episode Time'])
    training_history.to_csv('result-dataset/training_history.csv')

game_loop(alpha, gamma, epsilon_discount)