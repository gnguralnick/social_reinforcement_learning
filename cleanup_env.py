from __future__ import annotations
import random
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors
from ray.rllib.env import MultiAgentEnv

from agent import CleanupAgent
# from social-reinforcement-learning.agent import CleaupAgent


NearBy = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05
potential_waste_area = 6 * 25

class CleanupEnv(MultiAgentEnv):
    """
    Cleanup environment
    """

    def __init__(self, num_agents=5):
        """
        Initialise the environment.
        """
        self.num_agents = num_agents
        self.timestamp = 0

        self.action_space = Discrete(4)
        self.observation_space = Tuple((Box(low=0, high=25, shape=(self.num_agents, 2), dtype=np.int32), Box(low=-1, high=1, shape=(25, 18), dtype=np.int32)))
        self.agents = {}
        self.setup_agents()

        self.num_dirt = 0
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()
        self.map = np.zeros((25, 18))
        for i in range(0, 25, 2):
            for j in range(6):
                self.map[i][j] = -1
                self.num_dirt += 1

    def setup_agents(self):
        for i in range(self.num_agents):
            agent_id = str(i)
            spawn_point = [random.randint(0, 24), random.randint(0, 17)]
            while spawn_point[0] % 2 == 0 and spawn_point[1] < 6:
                # do not spawn on dirt
                spawn_point = [random.randint(0, 24), random.randint(0, 17)]
            agent = CleanupAgent(agent_id, spawn_point)
            self.agents[agent_id] = agent

    def reset(self, seed: int | None = None, options: dict = dict()) -> tuple:
        """
        Reset the environment.
        """
        # Set seed
        super().reset(seed=seed)
        self.timestamp = 0
        self.agents = {}
        self.setup_agents()
        self.num_dirt = 0
        self.map = np.zeros((25, 18))
        for i in range(0, 25, 2):
            for j in range(6):
                self.map[i][j] = -1
                self.num_dirt += 1

        observations = {}
        pos = np.zeros((self.num_agents, 2))
        a_keys = sorted(self.agents.keys())  # key must be sorted
        for agent_key in a_keys:
            pos[int(agent_key)] = self.agents[agent_key].pos
        for agent_key in a_keys:
            observations[self.agents[agent_key].agent_id] = (pos, self.map)
        return observations, {}

    def step(self, actions):
        """
        Take a step in the environment.
        """
        obs = {}
        rewards = {}
        dones = {}
        has_agent = set()
        self.timestamp += 1

        for agent in self.agents.values():
            action = actions[agent.agent_id]
            reward = 0
            if action == 0:  # up
                x, new_y = agent.pos[0], agent.pos[1]  # y is not exactly new
                new_x = x - 1 if x > 0 else x
                if (new_x, new_y) not in has_agent:
                    agent.pos = np.array([new_x, new_y])
                else:
                    new_x = x
                has_agent.add((new_x, new_y))
                reward += self.calculate_reward(new_x, new_y)
            elif action == 1:  # right
                new_x, y = agent.pos[0], agent.pos[1]
                new_y = y + 1 if y < 17 else y
                if (new_x, new_y) not in has_agent:
                    agent.pos = np.array([new_x, new_y])
                else:
                    new_y = y
                has_agent.add((new_x, new_y))
                reward += self.calculate_reward(new_x, new_y)
            elif action == 2:  # down
                x, new_y = agent.pos[0], agent.pos[1]
                new_x = x + 1 if x < 24 else x
                if (new_x, new_y) not in has_agent:
                    agent.pos = np.array([new_x, new_y])
                else:
                    new_x = x
                has_agent.add((new_x, new_y))
                reward += self.calculate_reward(new_x, new_y)
            elif action == 3:  # left
                new_x, y = agent.pos[0], agent.pos[1]
                new_y = y - 1 if y > 0 else y
                if (new_x, new_y) not in has_agent:
                    agent.pos = np.array([new_x, new_y])
                else:
                    new_y = y
                has_agent.add((new_x, new_y))
                reward += self.calculate_reward(new_x, new_y)
            rewards[agent.agent_id] = reward
            agent.reward += reward
        self.compute_probabilities()
        self.spawn_apples_and_waste(has_agent)
        pos = np.zeros((self.num_agents, 2))
        a_keys = sorted(self.agents.keys())  # key must be sorted
        for agent_key in a_keys:
            pos[int(agent_key)] = self.agents[agent_key].pos
        for agent_key in a_keys:
            obs[self.agents[agent_key].agent_id] = (pos, self.map)

        dones["__all__"] = self.timestamp == 1000
        return obs, rewards, dones, {"__all__": False}, {}

    def calculate_reward(self, x, y):
        if self.map[x][y] == -1:
            self.map[x][y] = 0
            self.num_dirt -= 1
            return 0
        if self.map[x][y] == 1:
            self.map[x][y] = 0
            return 1
        return 0

    def compute_probabilities(self):
        waste_density = 0
        if potential_waste_area > 0:
            waste_density = self.num_dirt / potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (1 - (waste_density - thresholdRestoration)
                              / (thresholdDepletion - thresholdRestoration)) \
                             * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def spawn_apples_and_waste(self, has_agent):
        # spawn apples, multiple can spawn per step
        for i in range(25):
            for j in range(12, 18, 1):
                rand_num = np.random.rand(1)[0]
                if rand_num < self.current_apple_spawn_prob and (i, j) not in has_agent:
                    self.map[i][j] = 1
        # spawn one waste point, only one can spawn per step
        if self.num_dirt < potential_waste_area:
            dirt_spawn = [random.randint(0, 24), random.randint(0, 5)]
            if self.map[dirt_spawn[0]][dirt_spawn[1]] != -1:
                rand_num = np.random.rand(1)[0]
                if rand_num < self.current_waste_spawn_prob and (dirt_spawn[0], dirt_spawn[1]) not in has_agent:
                    self.map[dirt_spawn[0]][dirt_spawn[1]] = -1
                    self.num_dirt += 1

    def render(self):
        """
        Render the environment.
        """
        labels = [[""]*18 for i in range(25)]
        for agent in self.agents.values():
            labels[agent.pos[0]][agent.pos[1]] += "{}({}) ".format(agent.agent_id, agent.reward)
        cmap = colors.ListedColormap(['tab:brown', 'white', 'green'])
        bounds = [-1, -0.5, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(6,6))
        plt.gca().invert_yaxis()
        plt.pcolor(self.map, cmap=cmap, edgecolors='k', linewidths=2)
        plt.show()

        return labels

