import torch
import torch.nn as nn

class UNetwork(nn.Module):
    def __init__(self, layers: list[tuple[int, int]], pref_embedding=1):
        super(UNetwork, self).__init__()
        self.pref_embedding = pref_embedding
        # self.coord1 = nn.Linear(2, 128)
        # self.coord2 = nn.Linear(128, 64)
        # self.coord2_1 = nn.Linear(6, 2)
        # self.coord3 = nn.Linear(64, self.pref_embedding)
        # self.leaky_relu = nn.LeakyReLU()
        # torch.nn.init.xavier_uniform_(self.coord1.weight)
        # torch.nn.init.xavier_uniform_(self.coord2.weight)
        # torch.nn.init.xavier_uniform_(self.coord2_1.weight)
        # torch.nn.init.xavier_uniform_(self.coord3.weight)

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(layers):
            self.layers.append(nn.Linear(in_dim, out_dim))
            nn.init.xavier_uniform_(self.layers[i].weight)

    def forward(self, coord):
        c = coord.view(coord.size(0), -1)
        #c = torch.nn.functional.normalize(c, dim=1)
        # c = self.leaky_relu(self.coord1(c))
        # c = self.leaky_relu(self.coord2(c))
        # # c = torch.relu(self.coord2_1(c))
        # c = self.coord3(c)
        for i, layer in enumerate(self.layers):
            c = layer(c)
            if i < len(self.layers) - 1:
                c = torch.relu(c)
        c = c.view(coord.size(0), self.pref_embedding)
        return c