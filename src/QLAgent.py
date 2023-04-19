import numpy as np
from src.snake import Snake
import random
import json

up = (0,-1)
down = (0,1)
left = (-1,0)
right = (1,0)

class QLearner:
    def __init__(self, display_width, display_height, grid_size):
        self.display_width = display_width
        self.disply_height = display_height
        self.grid_size = grid_size

        # Definition of the states:
        #       Current Directions: up, down, right, left
        #       Food positions relative to the snake: up, down, right, left
        #       Dangers around : up, down, right, left
        # Actions: Up, Down, right, and left
        # Consider taking the states as a binary variables, each will have 2,
        # then you will have 2^12 *4 = 16384 states
        # I'm going to define the dimension of the table as the following:
        # current_up: 0, 1
        # current_down: 0,1
        # current_right: 0, 1
        # current_left: 0, 1
        # food_up: 0, 1
        # food_down: 0, 1
        # food_right: 0, 1
        # food_left: 0, 1
        # danger_up: 0, 1
        # danger_down: 0, 1
        # danger_right: 0, 1
        # danger_left: 0, 1
        # actions: up, down, left, right

        self.Q_tables = np.zeros((2,2,2,2,2,2,2,2,2,2,2,2,4))   # Depends on the State and Actions

        # Q learning Parameters
        self.epsilon = 1.0
        self.lr = 0.01
        self.discount_rate = 0.95
        self.epsilon_discount = 0.9991
        self.min_epsilon = 0.001
        # State and Action for Q values
        self.history = []

        # The action space, from state to actions
        self.actions = {
            0: up,
            1: down,
            2: right,
            3: left
        }

    def get_action(self, state):
        # Get random action by exploiting
        if random.random() < self.epsilon:
            return random.choice([0,1,2,3])
        # Return matrix of available actions
        return np.argmax(self.Q_tables[state])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_discount, self.min_epsilon)

    # Reset the learner

    def update_Q_valeus(self, old_state, new_state, action, reward):
        self.Q_tables[old_state][action] = (1 - self.lr) * self.Q_tables[old_state][action] + \
                                           self.lr  * (reward + self.discount_rate * max(self.Q_tables[new_state]))


#   Q-Matrix for the agent
#   R-Matrix: Give rewards based on the city block distance
#   Definition:
#    - based on the state (distance between food and snake, direction)
