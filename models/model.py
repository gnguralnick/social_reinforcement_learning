import gymnasium as gym
from ray.rllib import MultiAgentEnv
from tensorflow.python import keras
from tensorflow.python.keras import layers


class Model:

    def __init__(self, obs_space: gym.spaces.Tuple, action_space: gym.spaces.Space, num_outputs, model_config, name):
        self._model = None
        self.obs_space = obs_space
        self.pos_space = obs_space[0]
        self.state_space = obs_space[1]
        self.action_space = action_space

        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name

        self.pos_input = layers.Input(shape=self.pos_space.shape, name="pos_input")
        self.state_input = layers.Input(shape=self.state_space.shape, name="state_input")

        self.use_model = True

    def build_model(self):
        if not self.use_model:
            return
        self._model = keras.Model()

    def predict(self, obs):
        if not self.use_model:
            raise NotImplementedError
        return self._model.predict(obs)

    def compile(self, optimizer='adam', loss='mse', metrics=None):
        if not self.use_model:
            return
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        if not self.use_model:
            print(f'Model: {self.name} - Model not used')
        self._model.summary()
