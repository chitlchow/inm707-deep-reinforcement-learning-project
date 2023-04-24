import numpy as np
import pygame
from src.snake import Snake
from src.food import Food
from src.DDQN_NoisyNet import NoisyNet_agent
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

num_episodes = 10000
game_speed = 1000

alpha = 0.001
gamma = 0.9
epsilon_discount = 0.9992

# Show graphics if set to True
graphics = False

# Main program for the game

def game_loop(alpha, gamma, epsilon_discount):
    learner = NoisyNet_agent(learning_rate=alpha, gamma=gamma, epsilon_decay=epsilon_discount)
    snake = Snake(screen_width, screen_height)
    food = Food(screen_width, screen_height)

    if graphics:
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((screen_width, screen_height))
        surface = pygame.Surface(screen.get_size())
        surface.convert()
        drawGrid(surface)

    training_histories = []
    high = 0
    for episode in range(1, num_episodes + 1):
        score = 0
        steps_without_food = 0
        start = time.time()

        learner.episode_history = []
        learner.reward_history = []
        learner.clear_episode_memories()
        game_over = False
        swap = False
        while not crash or steps_without_food == 1000:
            reward = 0
            swap = not swap

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
                if score >= high:
                    high = score
                    # Save the best model achieved new highest scores
                    learner.model1.save_model(f_name='noisy-model1-snapshot-score-{}.pth'.format(high))
                    learner.model2.save_model(f_name='noisy-model2-snapshot-score-{}.pth'.format(high))
                reward = 10

                # Reset the counter
                steps_without_food = 0
                food.randomize_position()
            else:
                steps_without_food += 1

            # Case where episodes is going to be terminated
            if crash or steps_without_food == 1000:
                # Break the loop if crashing or timeout
                reward = -10
                game_over = True


            # Update the Q-values

            learner.reward_history.append(reward)

            # Memorize step
            learner.memorize(current_state,
                             reward=reward,
                             action=action,
                             new_state=new_state,
                             game_over=int(game_over))


            # learner.update_priorities(learner.priorities, td_error)

            # learner.train_step(current_state, action, reward, new_state)
            learner.train_short_memories(current_state, action, reward, new_state, int(game_over), swap)

            if graphics:
                clock.tick(game_speed)
                drawGrid(surface)
                snake.draw(surface)
                food.draw(surface)
                screen.blit(surface, (0,0))
                pygame.display.update()

        learner.train_long_memories(swap)
        learner.clear_episode_memories()
        end = time.time()
        ep_time = end - start

        # Reset the environment
        learner.score_history.append(score)
        learner.ep_time_history.append(ep_time)
        learner.update_epsilon()
        reset_game(snake, food)
        # print("EP: {}, Score: {}".format(episode, score))
        if episode % 25 == 0:
            print("EP: {}, Mean Score: {:.2f}, epsilon: {:.4f}, episode time: {:.6f}, Highest: {}"\
                  .format(episode,
                          np.mean(np.array(learner.score_history)),
                          learner.epsilon,
                          np.mean(np.array(learner.ep_time_history)),
                          high))
            training_histories.append((
                episode, np.mean(np.array(learner.score_history)), learner.epsilon,
                np.mean(np.array(learner.ep_time_history)), high)
            )
            # Clear the history of last 25 episodes
            learner.score_history = []

    training_history = pd.DataFrame(training_histories, columns=['ep', 'mean_score_last25_ep', 'epsilon', 'ep_time', 'all_time_highest'])
    training_history.to_csv('result-dataset/training_history_ddqn_noisy.csv')

game_loop(alpha, gamma, epsilon_discount)