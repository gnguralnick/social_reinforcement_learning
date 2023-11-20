from models.centralized_q import CentralizedQNetwork
import torch
import numpy as np
from models.util import ReplayBuffer

import random


class CentralizedQAgent:
    def __init__(self, device, num_agents, action_size, state_dim, buffer_size=1000,
                 batch_size=128, lr=0.0001, epsilon=1.0, epsilon_decay=0.999, gamma=0.99, verbose=False):
        self.device = device
        
        self.num_agents = num_agents
        self.action_size = action_size
        self.batch_size = batch_size
        self.state_dim = state_dim

        self.q_network = CentralizedQNetwork(num_agents, action_size, state_dim, verbose=verbose).to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        if random.random() > max(self.epsilon, 0.05):
            return torch.argmax(q_values, dim=2).cpu().numpy()
        else:
            return np.random.choice(self.action_size, (1, self.num_agents))
        
    def step(self, state, action, reward, next_state):
        self.memory.add((state, action, reward, next_state))
        if len(self.memory) > self.batch_size:
            self.train()
        self.epsilon *= self.epsilon_decay

    def train(self):
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*experiences)

        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().unsqueeze(2).to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().unsqueeze(2).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)

        q_values = self.q_network(states)
        q_values = q_values.gather(2, actions)

        next_q_values = self.q_network(next_states)
        next_q_values = next_q_values.max(2)[0].unsqueeze(2)

        expected_q_values = rewards + self.gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, expected_q_values)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()