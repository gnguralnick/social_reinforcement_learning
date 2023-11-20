import torch
import torch.nn as nn

class CentralizedQNetwork(nn.Module):
    def __init__(self, num_agents, action_size, input_dim, verbose=False):
        super(CentralizedQNetwork, self).__init__()
        self.fc1 = nn.Linear(num_agents * input_dim, 256)
        self.fc2_1 = nn.Linear(256, 128)
        # self.fc2_2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_agents * action_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2_1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2_2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.leaky_relu = nn.LeakyReLU()
        # self.softmax = nn.Softmax()

        self.num_agents = num_agents
        self.action_size = action_size
        self.verbose = verbose

    def forward(self, state):
        if self.verbose:
            print("+++++++++++++++++++++++++++")
            print("Q Net Input:")
            print(state)
        x = state.view(state.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(self.fc2_1(x))
        # x = torch.relu(self.fc2_2(x))
        x = self.fc3(x).view(-1, self.num_agents, self.action_size)
        if self.verbose:
            print("Q Net Output")
            print(x)
            print("+++++++++++++++++++++++++++")
        return x