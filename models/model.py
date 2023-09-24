import gymnasium as gym
from tensorflow.python import keras
from tensorflow.python.keras import layers


class Model:

    def __init__(self, obs_space: gym.spaces.Tuple, action_space: gym.spaces.Space, num_outputs, model_config, name, build=False):
        self._model = None
        self.obs_space = obs_space
        self.pos_space = obs_space[0]
        self.state_space = obs_space[1]
        self.action_space = action_space
        self.num_actions = gym.spaces.flatten_space(self.action_space).shape[0]

        self.num_outputs = num_outputs
        self.model_config = model_config
        self.name = name

        self.pos_input = layers.Input(shape=self.pos_space.shape, name="pos_input")
        self.state_input = layers.Input(shape=self.state_space.shape, name="state_input")

        if build:
            self.build_model()
            self.use_model = True
        else:
            self.use_model = False

    def build_model(self):
        if not self.use_model:
            return
        self._model = keras.Model()

    def predict(self, obs, **kwargs):
        if not self.use_model:
            raise NotImplementedError  # this should always be overridden in subclasses where use_model is False
        return self._model.predict(obs, **kwargs)

    def compile(self, optimizer='adam', loss='mse', metrics=None):
        if not self.use_model:
            return
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        if not self.use_model:
            print(f'Model: {self.name} - Model not used')
        self._model.summary()

    def fit(self, *args, **kwargs):
        if not self.use_model:
            return
        return self._model.fit(*args, **kwargs)
