from models.model import Model
from tensorflow.python.keras import layers, Model as KerasModel

from models.common import build_q_arch


class CentralizedQModel(Model):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedQModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, build=True)
        self.use_model = True

    def build_model(self):
        last_hidden = build_q_arch(self.state_input, self.pos_input)

        actions = layers.Dense(self.num_actions * self.num_outputs, activation="linear")(last_hidden)
        action = layers.Reshape((self.num_outputs, self.num_actions))(actions)

        self._model = KerasModel(inputs=[self.pos_input, self.state_input], outputs=action)
