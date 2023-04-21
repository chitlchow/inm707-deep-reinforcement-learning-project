import torch
from torch import nn
import os
import numpy as np
import random
from collections import deque

up = (0,-1)
down = (0,1)
left = (-1,0)
right = (1,0)

# Set up for the model to use any tensor processing device
print(torch.has_mps)
device = torch.device('cpu')

class DQN_Agent():
    def __init__(self, learning_rate, gamma, epsilon_decay):
        # ANN model and parameters for training
        self.model = DQ_Network(12, 300, 3)
        self.model.to(device)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 0.7
        self.epsilon_decay = epsilon_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        # Short memories
        self.short_memories_size = 0
        self.short_memories = {
            "states": [],
            "rewards": [],
            "actions": [],
            "new_states": []
        }
        self.score_history = []
        self.reward_history = []
        self.actions = {
            # 0: up,
            # 1: down,
            # 2: right,
            # 3: left
            0: 'straight',
            1: 'turn_right',
            2: 'turn_left'
        }
        self.episode_history = []

    def train_step(self, current_state, action, reward, new_state):
        current_state = np.array(current_state)
        current_state = torch.from_numpy(current_state).type(torch.Tensor).to(device)
        new_state = np.array(new_state)
        new_state = torch.from_numpy(new_state).type(torch.Tensor).to(device)

        reward = torch.tensor(reward, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)


        q_current = self.model(current_state)
        target = q_current.clone()
        # print(q_current)
        q_new = reward + self.gamma * torch.max(self.model(new_state))
        # print(target[action])
        target[action] = q_new
        # print(q_new)
        # print(target)
        # print(target, q_current)
        self.optimizer.zero_grad()
        loss = self.loss_func(target, q_current)
        # print(loss)
        loss.backward()
        self.optimizer.step()


    def get_action(self, state):
        # Get random action by exploiting
        if random.random() < self.epsilon:
            return random.choice([0,1,2])
        # Return matrix of available actions
        else:
            state_vector = np.array(state)
            state_vector = torch.from_numpy(state_vector).type(torch.Tensor)
            # print(torch.argmax(self.model(state_vector)).item())
            return torch.argmax(self.model(state_vector)).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon* self.epsilon_decay, 0.001)


    def train_short_memories(self):
        current_states = np.array(self.short_memories['states'])
        current_states = torch.from_numpy(current_states).type(torch.Tensor)
        new_states = np.array(self.short_memories['new_states'])
        new_states = torch.from_numpy(new_states).type(torch.Tensor)
        rewards = np.array(self.short_memories['rewards'])
        rewards = torch.from_numpy(rewards).type(torch.Tensor)

        actions = np.array(self.short_memories['actions'])
        actions = torch.from_numpy(actions).type(torch.long)

        # print(actions.size(0))
        actions = actions.reshape(1,10)
        rewards = rewards.reshape(-1, 1)

        q_pred = self.model(current_states).gather(1, actions).reshape(-1,1)
        # This is a (10, 1) vector
        q_new = rewards + self.gamma * torch.max(self.model(new_states), dim=1, keepdim=True)[0]
        self.optimizer.zero_grad()
        loss = self.loss_func(q_pred, q_new)
        # print(loss)
        loss.backward()
        self.optimizer.step()

    def memorize(self, current_state, reward, action, new_state):
        # Only memorize if there's space
        new_memory = {
            'states': [current_state],
            'rewards': [reward],
            'actions': [action],
            'new_states': [new_state]
        }
        for key in new_memory:
            if key in self.short_memories:
                self.short_memories[key].extend(new_memory[key])
        self.short_memories_size += 1

    def clear_episode_history(self):
        self.episode_history = []
    def clear_memory(self):
        self.short_memories = {
            "states": [],
            "rewards": [],
            "actions": [],
            "new_states": []
        }
        self.short_memories_size = 0


class DQ_Network(nn.Module):
    # The network is to predict the Q-value - not the actions
    def __init__(self, input_dim, hidden_dim ,output_dim):
        super(DQ_Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.Softmax()
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        # out = self.softmax(out)
        # Returns the Q-value of each actions
        return out

    def save_model(self, f_name="model.pth"):
        model_folder_path = '../DRL_models'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        f_name = os.path.join(model_folder_path, f_name)
        torch.save(f_name)


