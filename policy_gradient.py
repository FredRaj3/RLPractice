import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

policy = Policy()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters(), lr = 0.01)

def update_policy(rewards, log_probs, optimizer):
    log_probs = torch.stack(log_probs)
    loss = -torch.mean(log_probs) * (sum(rewards) - 15)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

gamma = 0.99

for episode in range(5000):
    state, _ = env.reset()
    done = False
    score = 0
    log_probs = []
    rewards = []

    while not done:
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)

