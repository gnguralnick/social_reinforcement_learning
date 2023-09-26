from models.model import Model
import numpy as np


class RandomModel(Model):

    def predict(self, obs, **kwargs):
        # randomly sample self.num_outputs actions from self.action_space
        actions = np.zeros((1, self.num_outputs, self.num_actions))
        for i in range(self.num_outputs):
            action = self.env.action_space.sample()
            actions[0][str(i)][action] = 1
        return actions

