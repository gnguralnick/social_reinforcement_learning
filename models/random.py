from models.model import Model
import numpy as np


class RandomModel(Model):

    def predict(self, obs, **kwargs):
        # randomly sample self.num_outputs actions from self.action_space
        actions = [self.action_space.sample() for _ in range(self.num_outputs)]
        return np.array(actions)

