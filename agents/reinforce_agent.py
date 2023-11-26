from models import UNetwork
import torch
import numpy as np

class ReinforceUAgent:
    def __init__(self, device, num_action_outputs, action_size, buffer_size=10000, batch_size=64, lr=0.001, gamma=0.9999):
        self.num_action_outputs = num_action_outputs
        self.action_size = action_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma

        self.device = device

        self.value_network = UNetwork().to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=lr)

        self.policy_network = UNetwork().to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)


    def train(self, episode):
        returns = []
        for i, (state, action_log_probs, reward, next_state) in enumerate(episode):
            returns.append(sum([r * self.gamma ** i for i, (_, _, r, _) in enumerate(episode[i:])]))
        returns = torch.tensor(returns).float().to(self.device)

        states, action_log_probs, rewards, next_states = zip(*episode)

        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        action_log_probs = torch.from_numpy(np.vstack(action_log_probs)).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)

        current_values = self.value_network(states)
        next_values = self.value_network(next_states)
        expected_values = rewards + self.gamma * next_values

        value_loss = torch.nn.functional.mse_loss(current_values, expected_values)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        advantage = returns - current_values
        policy_loss = -advantage * action_log_probs
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        value = self.value_network(state)
        probs = self.policy_network(state)
        action = torch.multinomial(probs, num_samples=1)
        return action.cpu().detach().numpy(), value.cpu().detach().numpy()
    
    
