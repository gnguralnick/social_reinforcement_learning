from model import Model
from tensorflow.python.keras import layers, Model as KerasModel

from models.common import build_q_arch


class CentralizedQModel(Model):

    def build_model(self):
        last_hidden = build_q_arch(self.state_input, self.pos_input)

        actions = layers.Dense(self.action_space.shape * self.num_outputs, activation="linear")(last_hidden)
        action = layers.Reshape((self.num_outputs, self.action_space.shape))(actions)

        self._model = KerasModel(inputs=[self.pos_input, self.state_input], outputs=action)
