from abc import abstractmethod
import gymnasium as gym
import numpy as np
from ray.rllib import MultiAgentEnv
from tensorflow.python import keras
from tensorflow.python.keras import layers
import typing

from environments.env import ObjectiveEnv


class Model:

    def __init__(self, env: MultiAgentEnv, num_outputs, model_config, name, build=False):
        self._model = None
        self.env = env
        if not isinstance(self.env.observation_space, gym.spaces.Tuple):
            raise ValueError("Observation space must be a tuple of pos_space and state_space")
        self.obs_space = typing.cast(gym.spaces.Tuple, self.env.observation_space)
        self.pos_space = self.obs_space[0]
        self.state_space = self.obs_space[1]
        self.action_space = self.env.action_space
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
        raise NotImplementedError

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


class ObjectiveModel(Model):

    def __init__(self, env: ObjectiveEnv, num_outputs, model_config, name):
        super().__init__(env, num_outputs, model_config, name, False)
        self.env = env

    @abstractmethod
    def reassign_agent_objectives(self):
        """
        Reassigns objectives to agents.
        """
        raise NotImplementedError
    
    def predict(self, obs, **kwargs):
        self.reassign_agent_objectives()
        actions = np.zeros((1, self.num_outputs, self.num_actions))
        for id in self.env.get_agent_ids():
            action = self.env.get_greedy_action(self.env.agents[id])
            actions[0][int(id)][action] = 1
        return actions