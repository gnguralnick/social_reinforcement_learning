from model import Model
import numpy as np


class RandomModel(Model):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(RandomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.use_model = False

    def predict(self, obs):
        # randomly sample self.num_outputs actions from self.action_space
        actions = [self.action_space.sample() for _ in range(self.num_outputs)]
        return np.array(actions)

