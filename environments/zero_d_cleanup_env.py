from ray.rllib.env import MultiAgentEnv

import numpy as np

import random

class ZeroDCleanupEnv(MultiAgentEnv):
    """
    0-dimensional Cleanup environment. In this game, the agents must clean up the dirt from the river before apples can spawn.
    Agent reward is only given for eating apples, meaning the agents must learn to clean up the dirt first and
    must learn to balance their individual rewards with the collective goal of cleaning up the river.
    In this variant of the environment, there is no actual grid; agents are simply either pickers or cleaners and the
    amount of dirt cleaned or apples picked is determined probabilistically.
    """

    def __init__(self, agent_ids, num_agents=10, area=150, thresholdDepletion: float=0.4, thresholdRestoration: float=0, wasteSpawnProbability: float=0.5, appleRespawnProbability: float=0.05, dirt_multiplier=10, use_heuristic=False):
        """
        Initialise the environment.
        """
        self._agent_ids = set(agent_ids)
        self.num_agents = num_agents
        self.timestamp = 0

        self.dirt_end = 1
        self.area = area
        self.potential_waste_area = 150
        self.apple_start = 1

        self.num_dirt = 78
        self.num_apples = 0

        self.starting_apple_spawn_prob = appleRespawnProbability
        self.starting_waste_spawn_prob = wasteSpawnProbability
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability

        self.thresholdDepletion = thresholdDepletion
        self.thresholdRestoration = thresholdRestoration
        self.dirt_multiplier = dirt_multiplier

        self.compute_probabilities()

        self.total_apple_consumed = 0
        self.step_apple_consumed = 0
        self.epoch = 0

        self.use_heuristic = use_heuristic

        super().__init__()

    def reset(self, seed: int | None = None, options: dict = {}) -> tuple:
        """
        Reset the environment.
        """
        # Set seed
        super().reset(seed=seed)
        self.timestamp = 0

        self.num_dirt = 78
        self.num_apples = 0
        self.current_apple_spawn_prob = self.starting_apple_spawn_prob
        self.current_waste_spawn_prob = self.starting_waste_spawn_prob
        self.compute_probabilities()

        self.total_apple_consumed = 0
        self.step_apple_consumed = 0

        observations = {
            id: np.array([self.num_apples, self.num_dirt, 0, 0]) for id in self.get_agent_ids()
        }

        info = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": self.num_apples,
            "dirt": self.num_dirt,
            "picker": 0,
            "cleaner": 0
        }

        return observations, info

    def step(self, action_dict: dict = dict()) -> tuple:
        """
        Take a step in the environment.
        """

        observations = {}
        rewards = {}
        dones = {}
        
        self.timestamp += 1
        self.step_apple_consumed = 0

        if self.use_heuristic:
            agent_frequency_in_dirt = self.num_dirt / (self.num_apples + self.num_dirt)
            num_agents_to_be_assigned_to_dirt = round(self.num_agents * agent_frequency_in_dirt)
            agents = list(self.get_agent_ids())
            apple_agents = agents[num_agents_to_be_assigned_to_dirt:]
            dirt_agents = agents[:num_agents_to_be_assigned_to_dirt]

        else:
            apple_agents = [agent_id for agent_id in self.get_agent_ids() if action_dict[agent_id] == 0]
            dirt_agents = [agent_id for agent_id in self.get_agent_ids() if action_dict[agent_id] == 1]

        step_reward = self.step_reward_calculation(apple_agents)
        total_reward = sum(step_reward.values())
        self.step_apple_consumed = total_reward
        self.total_apple_consumed += total_reward
        self.step_dirt_calculation(dirt_agents)
        self.compute_probabilities()
        new_apple, new_dirt = self.spawn_apples_and_waste()

        observations = {
            id: np.array([self.num_apples, self.num_dirt, len(apple_agents), len(dirt_agents)]) for id in self.get_agent_ids()
        }
        rewards = step_reward
        dones["__all__"] = self.timestamp == 1000

        infos = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": self.num_apples,
            "dirt": self.num_dirt,
            "picker": len(apple_agents),
            "cleaner": len(dirt_agents)
        }
        return observations, rewards, dones, {"__all__": False}, infos

    def uniform_distribute(self, num_items, num_spots):
        if num_items > num_spots:
            raise ValueError("Cannot distribute more items than spots.")
        indices = random.sample(range(num_spots), num_items)
        distribution = [0] * num_spots
        for index in indices:
            distribution[index] = 1
        return distribution

    def step_reward_calculation(self, apple_agent_ids):
        reward = {
            id: 0 for id in self.get_agent_ids()
        }
        apple_agents = list(apple_agent_ids)
        num_apple_agents = len(apple_agents)
        d_apple = self.uniform_distribute(self.num_apples, self.area)
        d_picker = self.uniform_distribute(num_apple_agents, self.area)
        for i in range(len(d_apple)):
            if d_apple[i] == 1 and d_picker[i] == 1:
                reward[apple_agents.pop()] = 1
                self.num_apples -= 1
        return reward

    def step_dirt_calculation(self, dirt_agent_ids):
        reward = {
            id: 0 for id in self.get_agent_ids()
        }
        dirt_agents = list(dirt_agent_ids)
        num_dirt_agents = len(dirt_agents)
        d_dirt = self.uniform_distribute(self.num_dirt, self.area)
        d_cleaner = self.uniform_distribute(num_dirt_agents, self.area)
        for i in range(len(d_dirt)):
            if d_dirt[i] == 1 and d_cleaner[i] == 1:
                self.num_dirt -= 1
                reward[dirt_agents.pop()] = 1
        return reward

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = self.num_dirt / self.potential_waste_area
        if waste_density >= self.thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = self.starting_waste_spawn_prob
            if waste_density <= self.thresholdRestoration:
                self.current_apple_spawn_prob = self.starting_apple_spawn_prob
            else:
                spawn_prob = (1 - (waste_density - self.thresholdRestoration)
                              / (self.thresholdDepletion - self.thresholdRestoration)) \
                             * self.starting_apple_spawn_prob
                self.current_apple_spawn_prob = spawn_prob

    def spawn_apples_and_waste(self):
        # spawn apples, multiple can spawn per step
        new_apple, new_dirt = 0, 0
        for _ in range(self.area - self.num_apples): # only potentially spawn apples in empty spots
            rand_num = np.random.rand(1)[0]
            if rand_num < self.current_apple_spawn_prob and self.num_apples < self.area:
                self.num_apples += 1
                new_apple += 1

        # spawn one waste point, only one can spawn per step
        if self.num_dirt < self.potential_waste_area:
            rand_num = np.random.rand(1)[0]
            if rand_num < self.current_waste_spawn_prob:
                self.num_dirt += 1
                new_dirt += 1
        return new_apple, new_dirt