import random
from environments.one_d_cleanup_env import OneDCleanupEnv
from models import UNetwork
import torch
from agents.util import ReplayBuffer
import numpy as np

class OneDUCoordinator:
    def __init__(self, device, env: OneDCleanupEnv, num_agents, num_roles, u_layers: list[tuple[int, int]],  buffer_size=10000, batch_size=64, lr=0.001, gamma=0.9999, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05):
        self.num_agents = num_agents
        self.num_roles = num_roles
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.env = env

        self.device = device

        self.u_network = UNetwork(u_layers).to(self.device)
        self.u_optimizer = torch.optim.Adam(self.u_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.u_optimizer, step_size=100, gamma=0.6)

        self.memory = ReplayBuffer(buffer_size)

    def generate_roles(self):
        if random.random() < max(self.epsilon, self.epsilon_min):
            num_dirt_agents = random.randint(0, self.num_agents)
            return num_dirt_agents, self.num_agents - num_dirt_agents
        
        all_imm_rewards = []
        all_next_states = []
        for i in range(self.num_agents + 1):
            new_p = self.num_agents - i
            new_c = i
            assignments = self.env.get_greedy_assignments(new_p, new_c)
            actions = self.env.get_greedy_actions(assignments)
            observations, rewards = self.env.simulate_step(actions)
            future_state = observations["coordinator"]
            exp_imm_reward = rewards["coordinator"]
            all_imm_rewards.append(exp_imm_reward)
            future_state = torch.tensor(future_state).float().unsqueeze(0).to(self.device)
            all_next_states.append(future_state)
        all_next_states = torch.stack(all_next_states).float().to(self.device)
        all_pred_rewards = self.u_network(all_next_states).flatten()
        all_imm_rewards = torch.tensor(all_imm_rewards).float().to(self.device)
        all_future_rewards = all_imm_rewards + self.gamma * all_pred_rewards
        max_reward_dirt_agents = round(torch.argmax(all_future_rewards).item().real)

        return max_reward_dirt_agents, self.num_agents - max_reward_dirt_agents
    
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