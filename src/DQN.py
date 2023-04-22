import torch
from torch import nn
import os
import numpy as np
import random
from collections import deque

# Set up for the model to use any tensor processing device

device = torch.device('cpu')

class DQN_Agent():
    def __init__(self, learning_rate, gamma, epsilon_decay):
        # ANN model and parameters for training
        self.model = DQ_Network(11, 256, 3)
        self.target_model = DQ_Network(11, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 0.7
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        # Short memories
        self.short_memories_size = 0
        self.short_memories = deque(maxlen=10)
        #     {
        #     "states": [],
        #     "rewards": [],
        #     "actions": [],
        #     "new_states": [],
        #     "game_overs": []
        # }

        self.episode_memories = deque()
        self.score_history = []
        self.reward_history = []
        self.actions = {
            0: 'straight',
            1: 'turn_right',
            2: 'turn_left'
        }

    # def train_step(self, current_state, action, reward, new_state):
    #     current_state = np.array(current_state)
    #     current_state = torch.from_numpy(current_state).type(torch.Tensor).to(device)
    #     new_state = np.array(new_state)
    #     new_state = torch.from_numpy(new_state).type(torch.Tensor).to(device)
    #
    #     reward = torch.tensor(reward, dtype=torch.float).to(device)
    #     action = torch.tensor(action, dtype=torch.long).to(device)
    #     q_current = self.model(current_state)[action]
    #     q_next = torch.max(self.target_model(new_state))
    #
    #     q_expected = reward
    #     if reward != -10:
    #         q_expected = reward + self.gamma * q_next
    #     # print(target[action])
    #     q_next = q_expected
    #
    #     self.optimizer.zero_grad()
    #     loss = self.loss_func(q_next, q_current)
    #     loss.backward()
    #     self.optimizer.step()


    def get_action(self, state):
        # Get random action by exploiting
        if random.random() < self.epsilon:
            return random.choice([0,1,2])
        # Return matrix of available actions
        else:
            state_vector = np.array(state)
            state_vector = torch.from_numpy(state_vector).type(torch.Tensor).to(device)
            # print(torch.argmax(self.model(state_vector)).item())
            return torch.argmax(self.model(state_vector)).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon* self.epsilon_decay, self.min_epsilon)


    def train_short_memories(self, states, actions, rewards, next_states, game_overs):
        self.train(states, actions, rewards, next_states, game_overs)

    def train_long_memories(self):
        if len(self.episode_memories) > 1000:
            sample = random.sample(self.episode_memories, 1000)
        else:
            sample = self.episode_memories
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.train(states, actions, rewards, next_states, game_overs)

    def train(self, states, actions, rewards, next_states, game_overs):
        states = torch.tensor(states, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            rewards = torch.unsqueeze(rewards, 0)
            actions = torch.unsqueeze(actions, 0)
            game_overs = (game_overs, )


            # print(actions.size(0))
        q_pred = self.model(states)
        q_expected = q_pred.clone()
        # print(actions)
        for i in range(len(game_overs)):
                # This is a (10, 1) vector
            if game_overs != 1:
                q_new = rewards[i] + self.gamma * torch.max(self.model(next_states[i]))* (1 - game_overs[i])
            q_expected[i][actions[i]] = q_new
            # print(q_new)
        self.optimizer.zero_grad()
        loss = self.loss_func(q_pred, q_expected)
        # print(loss)
        loss.backward()
        self.optimizer.step()


    def memorize(self, current_state, action, reward, new_state, game_over):
        self.short_memories.append((current_state, reward, action, new_state, game_over))
        self.episode_memories.append((current_state, reward, action, new_state, game_over))


        # Only memorize if there's space
        # new_memory = {
        #     'states': [current_state],
        #     'rewards': [reward],
        #     'actions': [action],
        #     'new_states': [new_state],
        #     'game_overs': [game_over]
        # }
        #
        # # Check if memories full, if yes, than clear the earliest memories
        # if self.short_memories_size == 10:
        #     for key in self.short_memories:
        #         self.short_memories[key].pop(0)
        #     self.short_memories_size -= 1
        # # Add new memories to the end of the list
        # for key in new_memory:
        #     if key in self.short_memories:
        #         self.short_memories[key].extend(new_memory[key])
        #         self.episode_memories[key].extend(new_memory[key])
        # self.short_memories_size += 1

    def clear_memory(self):
        self.short_memories = deque(maxlen=10)
        self.short_memories_size = 0
    def clear_episode_memories(self):
        self.episode_memories = deque()

class DQ_Network(nn.Module):
    # The network is to predict the Q-value - not the actions
    def __init__(self, input_dim, hidden_dim ,output_dim):
        super(DQ_Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # self.linear3 = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.Softmax()
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = self.linear2(out)
        # out = self.linear3(out)
        # out = self.softmax(out)
        # Returns the Q-value of each actions
        return out

    def save_model(self, f_name="model.pth"):
        model_folder_path = '../DRL_models'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        f_name = os.path.join(model_folder_path, f_name)
        torch.save(f_name)


