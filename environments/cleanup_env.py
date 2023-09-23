from __future__ import annotations
import random
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors
from ray.rllib.env import MultiAgentEnv

from agents.cleanup_agent import CleanupAgent

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


class CleanupEnv(MultiAgentEnv):
    """
    Cleanup environment. In this game, the agents must clean up the dirt from the river before apples can spawn.
    Agent reward is only given for eating apples, meaning the agents must learn to clean up the dirt first and
    must learn to balance their individual rewards with the collective goal of cleaning up the river.
    """

    def __init__(self, num_agents=5, height=25, width=18):
        """
        Initialise the environment.
        """
        super().__init__()
        self.num_agents = num_agents
        self.timestamp = 0

        self.height = height
        self.width = width
        self.dirt_end = round((1 / 3) * self.width)
        self.potential_waste_area = self.dirt_end * self.height
        self.apple_start = round((2 / 3) * self.width)

        self.action_space = Discrete(4)  # directional movement
        self.observation_space = Tuple(
            (Box(low=0, high=self.height, shape=(self.num_agents, 2), dtype=np.int32),  # agent positions
                Box(low=-1, high=1, shape=(self.height, self.width), dtype=np.int32))  # map grid
        )
        self.agents = {}
        self.setup_agents()

        self.num_dirt = 0
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()
        self.map = np.zeros((self.height, self.width))
        for i in range(0, self.height, 2):
            for j in range(self.dirt_end):
                self.map[i][j] = -1
                self.num_dirt += 1

    def setup_agents(self):
        for i in range(self.num_agents):
            agent_id = str(i)
            spawn_point = [random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
            while spawn_point[0] % 2 == 0 and spawn_point[1] < self.dirt_end:
                # do not spawn on dirt
                spawn_point = [random.randint(0, self.height - 1), random.randint(0, self.width - 1)]
            agent = CleanupAgent(agent_id, spawn_point)
            self.agents[agent_id] = agent

    def reset(self, seed: int | None = None, options: dict = None) -> tuple:
        """
        Reset the environment.
        """
        options = options if options is not None else dict()
        # Set seed
        super().reset(seed=seed)
        self.timestamp = 0
        self.agents = {}
        self.setup_agents()
        self.num_dirt = 0
        self.map = np.zeros((self.height, self.width))
        for i in range(0, self.height, 2):
            for j in range(self.dirt_end):
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
                new_y = y + 1 if y < self.width - 1 else y
                if (new_x, new_y) not in has_agent:
                    agent.pos = np.array([new_x, new_y])
                else:
                    new_y = y
                has_agent.add((new_x, new_y))
                reward += self.calculate_reward(new_x, new_y)
            elif action == 2:  # down
                x, new_y = agent.pos[0], agent.pos[1]
                new_x = x + 1 if x < self.height - 1 else x
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
        if self.potential_waste_area > 0:
            waste_density = self.num_dirt / self.potential_waste_area
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
        for i in range(self.height):
            for j in range(self.apple_start, self.width, 1):
                rand_num = np.random.rand(1)[0]
                if rand_num < self.current_apple_spawn_prob and (i, j) not in has_agent:
                    self.map[i][j] = 1
        # spawn one waste point, only one can spawn per step
        if self.num_dirt < self.potential_waste_area:
            dirt_spawn = [random.randint(0, self.height - 1), random.randint(0, self.dirt_end)]
            while self.map[dirt_spawn[0]][dirt_spawn[1]] == -1:  # do not spawn on already existing dirt
                dirt_spawn = [random.randint(0, self.height - 1), random.randint(0, 5)]

            rand_num = np.random.rand(1)[0]
            if rand_num < self.current_waste_spawn_prob and (dirt_spawn[0], dirt_spawn[1]) not in has_agent:
                self.map[dirt_spawn[0]][dirt_spawn[1]] = -1
                self.num_dirt += 1

    def find_nearest_object_from_agent(self, agent_pos):
        x, y = agent_pos
        closest_x, closest_y, min_distance = -1, -1, float('inf')
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] in [1, -1] and abs(i - x) + abs(j - y) <= min_distance:
                    min_distance = abs(i - x) + abs(j - y)
                    closest_x, closest_y = i, j
        return np.array(closest_y, closest_x)
    
    def get_greedy_action(self, agent_pos):
        nearest_obj = self.find_nearest_object_from_agent(agent_pos)
        if agent_pos[0] == nearest_obj[0]:
            if nearest_obj[1] < agent_pos[1]:
                return 3
            return 1
        if agent_pos[0] > nearest_obj[0]:
            return 0
        return 2   

    def render(self):
        """
        Render the environment.
        """
        cmap = colors.ListedColormap(['tab:brown', 'white', 'green'])
        bounds = [-1, -0.5, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        # create discrete colormap
        plt.rcParams["figure.figsize"] = [10, 10]
        fig, ax = plt.subplots()
 
        for agent in self.agents.values():
            t = "{}({}) ".format(agent.agent_id, agent.reward)
            plt.text(agent.pos[1]-0.4, agent.pos[0], t, fontsize=8)
        ax.imshow(self.map, cmap=cmap, norm=norm)
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(-.5, self.width, 1))
        ax.set_yticks(np.arange(-.5, self.height, 1))
        # if not labels:
        plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        plt.show()
