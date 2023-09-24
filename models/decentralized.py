from model import Model
from common import build_q_arch
from tensorflow.python.keras import layers, Model as KerasModel


class DecentralizedModel(Model):

    def build_model(self):
        last_hidden = build_q_arch(self.state_input, self.pos_input)

        action = layers.Dense(self.action_space.shape, activation="linear")(last_hidden)

        self._model = KerasModel(inputs=[self.pos_input, self.state_input], outputs=action)
