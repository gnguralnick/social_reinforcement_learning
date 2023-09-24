from models.model import Model
from tensorflow.python.keras import layers, Model as KerasModel

from models.common import build_q_arch


class CentralizedHelper(Model):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, parent, index):
        super(CentralizedHelper, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.use_model = False
        self.parent = parent
        self.index = index

    def predict(self, obs):
        action = self.parent.get_child_action(self.index)
        if action is None:
            self.parent.predict(obs)
            action = self.parent.get_child_action(self.index)
        return action

    def summary(self):
        self.parent.summary()

    def fit(self, *args, **kwargs):
        return self.parent.fit(*args, **kwargs)


class CentralizedQModel(Model):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedQModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.use_model = True
        self.action_buffer = [None for _ in range(self.num_outputs)]
        self.children = [CentralizedHelper(obs_space, action_space, num_outputs, model_config, name + '-' + i, self, i) for i in range(num_outputs)]

    def build_model(self):
        last_hidden = build_q_arch(self.state_input, self.pos_input)

        actions = layers.Dense(self.action_space.shape * self.num_outputs, activation="linear")(last_hidden)
        action = layers.Reshape((self.num_outputs, self.action_space.shape))(actions)

        self._model = KerasModel(inputs=[self.pos_input, self.state_input], outputs=action)

    def predict(self, obs):
        actions = self._model.predict(obs)
        self.action_buffer = actions
        return actions

    def get_child_action(self, index):
        action = self.action_buffer[index]
        self.action_buffer[index] = None
        return action


def get_centralized(obs_space, action_space, num_outputs, model_config, name):
    parent = CentralizedQModel(obs_space, action_space, num_outputs, model_config, name)
    return parent.children
