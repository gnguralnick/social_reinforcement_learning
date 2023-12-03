from ray.rllib.env import MultiAgentEnv

import numpy as np
import torch
from scipy.special import comb

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

        super().__init__()

        self.dirt_end = 1
        self.area = area
        self.potential_waste_area = self.area
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
        # self.total_reward_by_agent = {id: 0 for id in self.get_agent_ids()}
        self.use_heuristic = use_heuristic

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
        self.total_reward_by_agent = {id: 0 for id in self.get_agent_ids()}

        observations = {
            id: np.array([self.num_apples, self.num_dirt, 0, 0]) for id in self.get_agent_ids()
        }

        info = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": self.num_apples,
            "dirt": self.num_dirt,
            "picker": 0,
            "cleaner": 0,
            # "total_reward_by_agent": self.total_reward_by_agent,
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
        total_reward = step_reward
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
            "cleaner": len(dirt_agents),
            # "total_reward_by_agent": self.total_reward_by_agent,
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
        reward = (self.num_apples * len(apple_agent_ids)) / self.area
        self.num_apples -= reward
        return reward

    def step_dirt_calculation(self, dirt_agent_ids):
        reward = (self.num_dirt * len(dirt_agent_ids)) / self.area
        self.num_dirt -= reward
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
        new_apple = (self.area - int(self.num_apples)) * self.current_apple_spawn_prob
        self.num_apples += new_apple
        new_apple = min(self.num_apples, self.area)
        
        

        # spawn one waste point, only one can spawn per step
        new_dirt = self.current_waste_spawn_prob
        self.num_dirt += new_dirt
        new_dirt = min(self.num_dirt, self.potential_waste_area)

        return new_apple, new_dirt
    
    def transition_P(self, s0, s1):
        delta_a = s0[0] - s1[0]
        delta_d = s0[1] - s1[1]
        p1 = float(comb(s0[0], delta_a)*comb(self.area-s0[0], s1[2]-delta_a)) / comb(self.area, s1[2])
        p2 = float(comb(s0[1], delta_d)*comb(self.area-s0[1], s1[3]-delta_d)) / comb(self.area, s1[3])
        p = p1 * p2
        return p
    
    def simulate_future_state(self, new_p, new_c):
        apple_left = self.num_apples - ((self.num_apples * new_p) / self.area)
        dirt_left = self.num_dirt - ((self.num_dirt * new_c) / self.area)
        cur_dirt_density = dirt_left / self.area
        if cur_dirt_density >= self.thresholdDepletion:
            exp_new_dirt = dirt_left
            exp_new_apple = apple_left
        else:
            exp_new_dirt = dirt_left + 0.5
            exp_new_apple = apple_left + (self.area-apple_left)*(1-(cur_dirt_density-self.thresholdRestoration)/(self.thresholdDepletion-self.thresholdRestoration)) * self.starting_apple_spawn_prob
        s_next = [exp_new_apple, exp_new_dirt, new_p, new_c]
        return s_next
    
    def get_immediate_reward(self, n_pickers):
        return (self.num_apples * n_pickers) / self.area