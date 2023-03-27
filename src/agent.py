import numpy as np
from snake import Snake

class Agent:
    def __init__(self):
        self.R_matrix = np.zeros((4, 4))
        self.gamma = 0.7
        self.alpha = 0.2
        self.action_set = ['up', 'down', 'left', 'right']




