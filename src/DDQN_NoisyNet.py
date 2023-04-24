import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import random
from collections import deque

# Set up for the model to use any tensor processing device

device = torch.device('cpu')

class NoisyNet_agent():
    def __init__(self, learning_rate, gamma, epsilon_decay):
        # ANN model and parameters for training
        self.model1 = NoisyNet(11, 256, 3)
        self.model2 = NoisyNet(11, 256, 3)
        self.model2.load_state_dict(self.model1.state_dict())
        self.learning_rate = learning_rate
        self.model1_optimizer = torch.optim.Adam(self.model1.parameters(), lr=self.learning_rate)
        self.model2_optimizer = torch.optim.Adam(self.model2.parameters(), lr=self.learning_rate)

        # Q learning parameters
        self.gamma = gamma
        self.epsilon = 0.1
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01

        self.loss_func = nn.MSELoss()


        # Short memories
        self.short_memories_size = 0
        self.short_memories = deque(maxlen=10)
        self.episode_memories = deque()

        # Histories
        self.ep_time_history = []
        self.score_history = []
        self.reward_history = []
        self.actions = {
            0: 'straight',
            1: 'turn_right',
            2: 'turn_left'
        }

    def get_action(self, state, swap=False):
        # Get random action by exploiting
        if random.random() < self.epsilon:
            return random.choice([0,1,2])
        # Return matrix of available actions
        else:
            state_vector = np.array(state)
            state_vector = torch.from_numpy(state_vector).type(torch.Tensor).to(device)
            if swap:
            # print(torch.argmax(self.model(state_vector)).item())
                return torch.argmax(self.model2(state_vector)).item()
            return torch.argmax(self.model1(state_vector)).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon* self.epsilon_decay, self.min_epsilon)


    def train_short_memories(self, states, actions, rewards, next_states, game_overs, swap):
        self.train(states, actions, rewards, next_states, game_overs, swap)

    def train_long_memories(self, swap):
        if len(self.episode_memories) > 1000:
            sample = random.sample(self.episode_memories, 1000)
        else:
            sample = self.episode_memories
        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.train(states, actions, rewards, next_states, game_overs, swap)

    def train(self, states, actions, rewards, next_states, game_overs, swap):
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

        if swap:
            q_pred = self.model2(states)
        else:
            q_pred = self.model1(states)

        q_expected = q_pred.clone()
        for i in range(len(game_overs)):
                # This is a (10, 1) vector
            if game_overs != 1:
                if swap:
                    # Evaluate by model 1
                    q_new = rewards[i] + self.gamma * torch.max(self.model1(next_states[i]))* (1 - game_overs[i])
                else:
                    # Evaluate by model 2
                    q_new = rewards[i] + self.gamma * torch.max(self.model2(next_states[i])) * (1 - game_overs[i])
            q_expected[i][actions[i]] = q_new

        if swap:
            self.model2_optimizer.zero_grad()
        else:
            self.model1_optimizer.zero_grad()
        loss = self.loss_func(q_pred, q_expected)
        # print(loss)
        loss.backward()
        if swap:
            self.model2_optimizer.step()
        else:
            self.model1_optimizer.step()


    def memorize(self, current_state, action, reward, new_state, game_over):
        self.short_memories.append((current_state, action, reward, new_state, game_over))
        self.episode_memories.append((current_state, action, reward, new_state, game_over))

    def clear_memory(self):
        self.short_memories = deque(maxlen=10)
        self.short_memories_size = 0
    def clear_episode_memories(self):
        self.episode_memories = deque()

class NoisyNet(nn.Module):
    # The network is to predict the Q-value - not the actions
    def __init__(self, input_dim, hidden_dim ,output_dim):
        super(NoisyNet, self).__init__()
        self.layer1 = NoisyLinear(input_dim, hidden_dim)
        self.layer2 = NoisyLinear(hidden_dim, output_dim)

    def forward(self, x):
        out = torch.relu(self.layer1(x))
        out = self.layer2(out)
        # Returns the Q-value of each actions
        return out

    def save_model(self, f_name="model.pth"):
        model_folder_path = 'DDQN_models'
        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)
        f_name = os.path.join(model_folder_path, f_name)
        torch.save(self.state_dict(), f_name)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        sigma_init = self.sigma_init / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init)

    def forward(self, x):
        weight_eps = torch.empty_like(self.weight_sigma).normal_()
        bias_eps = torch.empty_like(self.bias_sigma).normal_()
        weight = self.weight_mu + self.weight_sigma * weight_eps
        bias = self.bias_mu + self.bias_sigma * bias_eps
        return F.linear(x, weight, bias)