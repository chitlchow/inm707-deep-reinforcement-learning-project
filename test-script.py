import torch
from torch import nn
from src.DQN import DQN_Agent, DQ_Network
import numpy as np

model = DQ_Network(12, 256, 4)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
gamma = 0.95

current_state = np.array([(0, 0, 0, 1, 1, 1, 1,1, 1, 1, 0, 1)])
new_state = np.array([(1, 0, 1, 1, 1, 1, 1,1, 1, 1, 0, 1)])
reward = 1
action = 2

current_state = torch.from_numpy(current_state).type(torch.Tensor)[0]
new_state = torch.from_numpy(new_state).type(torch.Tensor)[0]


for i in range(1, 10):
    Q_current = model(current_state)
    target = Q_current.clone()
    Q_new = reward + gamma * torch.max(model(new_state))
    print(Q_current)
    print(Q_new)

    target[action] = Q_new
    print(target)

    optimizer.zero_grad()
    loss = loss_func(target, Q_current)
    print(loss)
    loss.backward()
    optimizer.step()




