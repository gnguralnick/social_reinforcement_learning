from ray.rllib.env import MultiAgentEnv

import numpy as np
import torch

from agents import GreedyCleanUpAgent

import random

from enum import Enum

class CleanupRegion(Enum):
    APPLE = 1
    WASTE = -1

class OneDCleanupEnv(MultiAgentEnv):
    """
    1-dimensional Cleanup environment. In this game, the agents must clean up the dirt from the river before apples can spawn.
    Agent reward is only given for eating apples, meaning the agents must learn to clean up the dirt first and
    must learn to balance their individual rewards with the collective goal of cleaning up the river.
    In this variant of the environment, the areas containing dirt and apples are one-dimensional.
    Agents can move up and down within their area, and can cross over to the other area.
    """

    def __init__(self, agent_ids, num_agents=10, area=150, thresholdDepletion: float=0.4, thresholdRestoration: float=0, wasteSpawnProbability: float=0.5, appleRespawnProbability: float=0.05, dirt_multiplier=10, use_heuristic=False):
        """
        Initialise the environment.
        """
        self._agent_ids = set(agent_ids)
        self._agents = {id: GreedyCleanUpAgent(id, (0, 0), CleanupRegion.APPLE) for id in self._agent_ids}
        self.num_agents = num_agents
        self.timestamp = 0

        super().__init__()

        self.dirt_end = 1

        self.potential_apple_area = area
        self.apple_map = np.zeros(self.potential_apple_area)
        self.apple_agent_map = np.zeros(self.potential_apple_area)
        self.potential_waste_area = area
        self.waste_map = np.zeros(self.potential_waste_area)
        self.waste_agent_map = np.zeros(self.potential_waste_area)

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

    def reset(self, seed: int | None = None, options: dict = {}) -> tuple:
        """
        Reset the environment. Distribute agents uniformly across the two areas.
        """
        # Set seed
        super().reset(seed=seed)
        self.timestamp = 0

        self.apple_map = np.zeros(self.potential_apple_area)
        self.apple_agent_map = np.zeros(self.potential_apple_area)
        self.waste_map = np.zeros(self.potential_waste_area)
        self.waste_agent_map = np.zeros(self.potential_waste_area)

        self.num_dirt = 78
        self.num_apples = 0

        self.current_apple_spawn_prob = self.starting_apple_spawn_prob
        self.current_waste_spawn_prob = self.starting_waste_spawn_prob
        self.compute_probabilities()

        self.total_apple_consumed = 0
        self.step_apple_consumed = 0
        self.total_reward_by_agent = {id: 0 for id in self.get_agent_ids()}

        # Distribute agents uniformly across the two areas
        num_agents = len(self.get_agent_ids())
        apple_agents = num_agents // 2
        dirt_agents = num_agents - apple_agents
        apple_agent_ids = random.sample(self.get_agent_ids(), apple_agents)
        dirt_agent_ids = list(set(self.get_agent_ids()) - set(apple_agent_ids))

        for i, id in enumerate(apple_agent_ids):
            loc = (i // apple_agents) * self.potential_apple_area
            self._agents[id].region = CleanupRegion.APPLE
            self._agents[id].pos = np.array(loc)
            self.apple_agent_map[loc] = id
        
        for i, id in enumerate(dirt_agent_ids):
            loc = (i // dirt_agents) * self.potential_waste_area
            self._agents[id].region = CleanupRegion.WASTE
            self._agents[id].pos = np.array(loc)
            self.waste_agent_map[loc] = id

        for i in range(self.num_dirt):
            remaining_locs = np.where(self.waste_map == 0)[0]
            if len(remaining_locs) == 0:
                self.num_dirt = i
                break
            loc = random.choice(remaining_locs)
            self.waste_map[loc] = 1

        observations = {
            #id: np.array([self.num_apples, self.num_dirt, 0, 0]) for id in self.get_agent_ids()
            id: (self.num_apples, self.num_dirt, 0, 0, self.apple_map, self.waste_map) for id in self.get_agent_ids()
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
    
    def step(self, action_dict: dict[str, tuple[int, int]] = dict()) -> tuple:
        """
        Take a step in the environment.
        """

        observations = {}
        rewards = {}
        dones = {}
        dones["_all__"] = False

        self.timestamp += 1
        self.step_reward = 0

        num_pickers = 0
        num_cleaners = 0

        # Move agents
        for id, action in action_dict.items():
            region, direction = action
            agent = self._agents[id]

            if region != agent.region:
                reward = self.switch_region(id, region)
                if region == CleanupRegion.APPLE:
                    num_pickers += 1
                else:
                    num_cleaners += 1
            elif region == CleanupRegion.APPLE:
                num_pickers += 1
                reward = self.move_agent(id, direction)
            else:
                num_cleaners += 1
                reward = self.move_agent(id, direction)
            rewards[id] = reward
            self.step_reward += reward
            self.total_apple_consumed += reward

        self.compute_probabilities()
        self.spawn_apples_and_waste()

        observations = {
            id: (self.num_apples, self.num_dirt, num_pickers, num_cleaners, self.apple_map, self.waste_map) for id in self.get_agent_ids()
        }

        info = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": self.num_apples,
            "dirt": self.num_dirt,
            "picker": num_pickers,
            "cleaner": num_cleaners,
        }

        return observations, rewards, dones, info
    
    def switch_region(self, id, region):
        """
        Switch an agent's region.
        """
        agent = self._agents[id]
        
        if region == CleanupRegion.APPLE:
            self.apple_agent_map[agent.pos] = 0
            self.waste_agent_map[agent.pos] = id
            if self.apple_map[agent.pos] != 0:
                self.apple_map[agent.pos] = 0
                self.num_apples -= 1
                return 1 # reward for eating an apple
            agent.region = CleanupRegion.APPLE
        else:
            self.apple_agent_map[agent.pos] = id
            self.waste_agent_map[agent.pos] = 0
            if self.waste_map[agent.pos] != 0:
                self.waste_map[agent.pos] = 0
                self.num_dirt -= 1
                return 0 # no reward for cleaning dirt
            agent.region = CleanupRegion.WASTE

        return 0
    
    def move_agent(self, id, direction):
        """
        Move an agent.
        """
        agent = self._agents[id]
        map = self.apple_map if agent.region == CleanupRegion.APPLE else self.waste_map
        new_pos = agent.pos + direction
        if new_pos < 0 or new_pos >= len(map):
            return 0
        agent.pos = new_pos

        if map[new_pos] != 0:
            map[new_pos] = 0
            if agent.region == CleanupRegion.APPLE:
                self.num_apples -= 1
                return 1
            else:
                self.num_dirt -= 1
                return 0
        return 0

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
        for x in range(self.potential_apple_area):
            rand_num = np.random.rand(1)[0]
            if rand_num < self.current_apple_spawn_prob and self.apple_agent_map[x] == 0 and self.apple_map[x] == 0:
                self.apple_map[x] = 1
                self.num_apples += 1

        # spawn one waste point, only one can spawn per step        
        num_cleaners = len([agent for agent in self._agents.values() if agent.region == CleanupRegion.WASTE])
        if self.num_dirt + num_cleaners < self.potential_waste_area:
            rand_num = np.random.rand(1)[0]
            if rand_num < self.current_waste_spawn_prob:
                remaining_locs = np.where(self.waste_map == 0 and self.waste_agent_map == 0)[0]
                if len(remaining_locs) > 0:
                    loc = random.choice(remaining_locs)
                    self.waste_map[loc] = 1
                    self.num_dirt += 1

    def closest_objective(self, region, pos, map_override=None):
        """
        Returns a tuple (u, d) where u is the distance to the closest apple above the position and d is the distance to the closest apple below the position.
        """
        if map_override is None:
            map = self.apple_map if region == CleanupRegion.APPLE else self.waste_map
        else:
            map = map_override

        u = np.where(map[:pos] == 1)[0]
        u = -1 if len(u) == 0 else pos - u[-1]
        d = np.where(map[pos:] == 1)[0]
        d = -1 if len(d) == 0 else d[0] + 1
        return u, d
    
    def get_immediate_reward(self, agent: GreedyCleanUpAgent, action: tuple[CleanupRegion, int], map_override=None, update_map=False):
        """
        Returns the immediate reward for an agent performing an action.
        """
        role, direction = action
        if role == CleanupRegion.WASTE:
            # no reward for cleaning dirt
            return 0
        
        map = map_override if map_override is not None else self.apple_map
        
        if role != agent.region and map[agent.pos] != 0:
            # switching regions and there is an apple at the agent's position in the new region
            if update_map:
                map[agent.pos] = 0
            return 1
        
        u, d = self.closest_objective(agent.region, agent.pos)
        if u == 1 and direction == -1:
            # moving up and there is an apple above
            if update_map:
                map[agent.pos - 1] = 0
            return 1
        if d == 1 and direction == 1:
            # moving down and there is an apple below
            if update_map:
                map[agent.pos + 1] = 0
            return 1
        return 0
    
    def get_immediate_rewards(self, actions: dict[str, tuple[CleanupRegion, int]]):
        """
        Returns a list of immediate rewards for each agent performing an action.
        """
        rewards = {}
        # copy apple map so that we can update it without affecting the original
        # this ensures we don't allow agents to eat the same apple in the simulation
        apple_map = self.apple_map.copy()
        for id, action in actions.items():
            agent = self._agents[id]
            rewards[id] = self.get_immediate_reward(agent, action, map_override=apple_map, update_map=True)
    
    def simulate_future_state(self, actions: dict[str, tuple[CleanupRegion, int]]):
        """
        Simulate the future state of the environment after all agents perform their actions.
        """
        # copy apple map so that we can update it without affecting the original
        # this ensures we don't allow agents to eat the same apple in the simulation
        apple_map = self.apple_map.copy()
        waste_map = self.waste_map.copy()
        num_pickers = 0
        num_cleaners = 0
        for id, action in actions.items():
            agent = self._agents[id]
            if action[0] == CleanupRegion.APPLE:
                num_pickers += 1
            else:
                num_cleaners += 1
            self.get_immediate_reward(agent, action, map_override=apple_map, update_map=True)
            self.get_immediate_reward(agent, action, map_override=waste_map, update_map=True)
        num_apples = np.count_nonzero(apple_map)
        num_dirt = np.count_nonzero(waste_map)
        
        observations = {
            id: (num_apples, num_dirt, num_pickers, num_cleaners, apple_map, waste_map) for id in self.get_agent_ids()
        }
        return observations