import math
import random
from environments.zero_d_cleanup_env import ZeroDCleanupEnv
from models import UNetwork
import torch
from agents.util import ReplayBuffer
import numpy as np

class ZeroDUCoordinator:
    def __init__(self, device, num_action_outputs, action_size, u_layers: list[tuple[int, int]], buffer_size=10000, batch_size=64, lr=0.001, gamma=0.9999, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05):
        self.num_action_outputs = num_action_outputs
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = device

        self.u_network = UNetwork(u_layers).to(self.device)
        self.u_optimizer = torch.optim.Adam(self.u_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.u_optimizer, step_size=100, gamma=0.6)

        self.memory = ReplayBuffer(buffer_size)
    
    def value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return self.u_network(state).cpu().detach().numpy()
    
    def act(self, env: ZeroDCleanupEnv):
        if random.random() < max(self.epsilon, self.epsilon_min):
            actions = np.random.choice(self.action_size, (1, self.num_action_outputs))
            return actions
        
        all_imm_rewards = []
        all_next_states = []
        for i in range(self.num_action_outputs + 1):
            new_p = self.num_action_outputs - i
            new_c = i
            exp_imm_reward = env.get_immediate_reward(new_p)
            all_imm_rewards.append(exp_imm_reward)
            future_state = env.simulate_future_state(new_p, new_c)
            future_state = torch.tensor(future_state).float().unsqueeze(0).to(self.device)
            all_next_states.append(future_state)
        all_next_states = torch.stack(all_next_states).float().to(self.device)
        all_pred_rewards = self.u_network(all_next_states).flatten()
        all_imm_rewards = torch.tensor(all_imm_rewards).float().to(self.device)
        all_future_rewards = all_imm_rewards + self.gamma * all_pred_rewards
        max_reward_dirt_agents = round(torch.argmax(all_future_rewards).item().real)
        outs = np.zeros((1, self.num_action_outputs))
        for i in range(max_reward_dirt_agents):
            outs[0][i] = 1

        return outs
    
    def step(self, state, reward, next_state):
        self.memory.add((state, reward, next_state))
        self.epsilon *= self.epsilon_decay
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.train(experiences)

    def train(self, experiences):
        states, rewards, next_states = zip(*experiences)

        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)

        current_values = self.u_network(states)
        next_values = self.u_network(next_states)
        expected_values = rewards + self.gamma * next_values

        loss = torch.nn.functional.mse_loss(current_values, expected_values)
        self.u_optimizer.zero_grad()
        loss.backward()
        self.u_optimizer.step()