from models.model import Model
from models.common import build_q_arch
from tensorflow.python.keras import layers, Model as KerasModel


class DecentralizedModel(Model):

    def __init__(self, env, num_outputs, model_config, name):
        super(DecentralizedModel, self).__init__(env, num_outputs, model_config, name, build=True)

    def build_model(self):
        last_hidden = build_q_arch(self.state_input, self.pos_input)

        action = layers.Dense(self.num_actions, activation="linear")(last_hidden)

        self._model = KerasModel(inputs=[self.pos_input, self.state_input], outputs=action)
