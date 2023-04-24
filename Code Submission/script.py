import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_snake
import gym

# print(gym.version)
class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_size, output_size, replay_buffer_capacity=10000, batch_size=64,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, target_update_frequency=10):
        self.input_size = input_size
        self.output_size = output_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_frequency = target_update_frequency
        self.replay_buffer = []
        self.dqn = DQNNetwork(self.input_size, self.output_size)
        self.target_dqn = DQNNetwork(self.input_size, self.output_size)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.loss_criterion = nn.MSELoss()

    def update_replay_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.dqn(state_tensor)
                action = q_values.argmax().item()
            return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.FloatTensor([transition[0] for transition in batch])
        action_batch = torch.LongTensor([transition[1] for transition in batch])
        reward_batch = torch.FloatTensor([transition[2] for transition in batch])
        next_state_batch = torch.FloatTensor([transition[3] for transition in batch])
        done_batch = torch.FloatTensor([transition[4] for transition in batch])

        q_values = self.dqn(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_dqn(next_state_batch).max(1)[0]
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        loss = self.loss_criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_dqn(self):
        if self.target_update_frequency > 0 and len(self.replay_buffer) % self.target_update_frequency == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

    def train_dqn(self, num_episodes, max_steps_per_episode):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            for step in range(max_steps_per_episode):
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update_replay_buffer(state, action, reward, next_state, done)
                self.train()
                state = next_state
                total_reward += reward
                if done:
                    break
            self.update_target_dqn()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print("Episode: {}/{} | Total Reward: {:.2f} | Epsilon: {:.2f}".format(
                episode + 1, num_episodes, total_reward, self.epsilon))

    # Create Snake environment
env = gym.make('Snake')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Initialize DQN agent
dqn_agent = DQNAgent(input_size, output_size)

 # Train DQN agent
num_episodes = 1000
max_steps_per_episode = 1000
dqn_agent.train_dqn(num_episodes, max_steps_per_episode)

