from ray.rllib.env import MultiAgentEnv

import numpy as np

from agents import GreedyCleanUpAgent

import random

from enum import Enum

import matplotlib.pyplot as plt

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

    def __init__(self, agent_ids, num_agents=10, area=150, thresholdDepletion: float=0.4, thresholdRestoration: float=0, wasteSpawnProbability: float=0.5, appleRespawnProbability: float=0.05, dirt_multiplier=10):
        """
        Initialise the environment.
        """

        self._agent_ids = set(agent_ids)
        self._agents = {id: GreedyCleanUpAgent(id, 0, CleanupRegion.WASTE) for id in self._agent_ids}
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

        self.num_apples = 0
        self.num_dirt = 78
        self.num_pickers = 0
        self.num_cleaners = self.num_agents

        self.starting_apple_spawn_prob = appleRespawnProbability
        self.starting_waste_spawn_prob = wasteSpawnProbability

        self.thresholdDepletion = thresholdDepletion
        self.thresholdRestoration = thresholdRestoration
        self.dirt_multiplier = dirt_multiplier

        self.total_apple_consumed = 0
        self.step_apple_consumed = 0
        self.epoch = 0

    def reset(self, seed=None, options: dict = {}) -> tuple:
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

        self.num_apples = 0
        self.num_dirt = 78
        self.num_pickers = 0
        self.num_cleaners = self.num_agents

        self.total_apple_consumed = 0
        self.step_apple_consumed = 0
        self.total_reward_by_agent = {a_id: 0 for a_id in self.get_agent_ids()}

        # Distribute agents uniformly across the dirt area
        dirt_agent_ids = sorted(list(set(self.get_agent_ids())))

        dirt_init_locations = np.random.choice(self.potential_waste_area, size=self.num_dirt, replace=False)
        for i in dirt_init_locations:
            self.waste_map[i] = 1

        for i, a_id in enumerate(dirt_agent_ids):
            loc = int((i / self.num_cleaners) * self.potential_waste_area)
            self._agents[a_id].region = CleanupRegion.WASTE
            self._agents[a_id].pos = loc
            self.waste_agent_map[loc] = a_id
            if self.waste_map[loc] == 1:  # if there is dirt on the spot
                self.num_dirt -= 1
                self.waste_map[loc] = 0

        observations: dict[str, tuple] = {
            'coordinator': (self.num_apples, self.num_dirt, self.num_pickers, self.num_cleaners),
        }
        for a_id in self.get_agent_ids():
            agent = self._agents[a_id]
            closest_objective = self.closest_objective(agent.region, agent.pos)
            closest_agents = self.closest_agents(agent.region, agent.pos)
            observations[a_id] = (closest_objective[0], closest_objective[1], closest_agents[0], closest_agents[1])

        info = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": self.num_apples,
            "dirt": self.num_dirt,
            "picker": self.num_pickers,
            "cleaner": self.num_cleaners,
            "apple_map": self.apple_map,
            "waste_map": self.waste_map,
            "apple_agent_map": self.apple_agent_map,
            "waste_agent_map": self.waste_agent_map,
        }

        return observations, info

    def step(self, action_dict: dict[str, tuple[CleanupRegion, int]]) -> tuple:
        """
        Take a step in the environment.
        """
        observations = {}
        rewards = {}
        dones = {}
        dones["__all__"] = False

        self.timestamp += 1
        self.step_reward = 0

        rewards, self.num_apples, self.num_dirt, self.num_pickers, self.num_cleaners = self.perform_step(action_dict)
        reward = sum(rewards.values())
        self.step_reward += reward
        self.total_apple_consumed += reward

        observations = {
            'coordinator': (self.num_apples, self.num_dirt, self.num_pickers, self.num_cleaners),
        }
        for id in self.get_agent_ids():
            agent = self._agents[id]
            closest_objective = self.closest_objective(agent.region, agent.pos)
            closest_agents = self.closest_agents(agent.region, agent.pos)
            observations[id] = (closest_objective[0], closest_objective[1], closest_agents[0], closest_agents[1])

        info = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": self.num_apples,
            "dirt": self.num_dirt,
            "picker": self.num_pickers,
            "cleaner": self.num_cleaners,
            "apple_map": self.apple_map,
            "waste_map": self.waste_map,
            "apple_agent_map": self.apple_agent_map,
            "waste_agent_map": self.waste_agent_map,
        }

        if self.timestamp == 1000:
            dones["__all__"] = True

        return observations, rewards, dones, {"__all__": False}, info

    def perform_step(self, action_dict: dict[str, tuple[CleanupRegion, int]], apple_map=None, waste_map=None, apple_agent_map=None, waste_agent_map=None) -> tuple:
        if apple_map is None:
            apple_map = self.apple_map
        if waste_map is None:
            waste_map = self.waste_map
        if apple_agent_map is None:
            apple_agent_map = self.apple_agent_map
        if waste_agent_map is None:
            waste_agent_map = self.waste_agent_map

        # Move agents
        rewards = {}
        num_dirt = self.num_dirt
        num_apples = self.num_apples
        num_pickers = 0
        num_cleaners = 0
        for id, action in action_dict.items():
            region, direction = action
            agent = self._agents[id]

            if region != agent.region:
                apples_consumed, dirt_consumed = self.switch_region(id, region, apple_map, waste_map, apple_agent_map, waste_agent_map)
                num_apples -= apples_consumed
                num_dirt -= dirt_consumed
                reward = apples_consumed
                if agent.region == CleanupRegion.APPLE:
                    num_pickers += 1
                else:
                    num_cleaners += 1
            elif region == CleanupRegion.APPLE:
                num_pickers += 1
                apples_consumed, dirt_consumed = self.move_agent(id, direction, apple_map, waste_map, apple_agent_map, waste_agent_map)
                reward = apples_consumed
                num_apples -= apples_consumed
                num_dirt -= dirt_consumed
            else:
                num_cleaners += 1
                apples_consumed, dirt_consumed = self.move_agent(id, direction, apple_map, waste_map, apple_agent_map, waste_agent_map)
                num_apples -= apples_consumed
                num_dirt -= dirt_consumed
                reward = 0
            rewards[id] = reward


        current_apple_spawn_prob, current_waste_spawn_prob = self.compute_probabilities(num_dirt)
        num_apples_spawned, num_waste_spawned = self.spawn_apples_and_waste(num_dirt, num_cleaners, current_apple_spawn_prob, current_waste_spawn_prob, apple_map, waste_map, apple_agent_map, waste_agent_map)

        num_apples += num_apples_spawned
        num_dirt += num_waste_spawned

        return rewards, num_apples, num_dirt, num_pickers, num_cleaners

    def switch_region(self, id, region, apple_map, waste_map, apple_agent_map, waste_agent_map):
        """
        Switch an agent's region.
        Returns a tuple (r, d) where r is the number of apples eaten and d is the number of dirt cleaned.
        """
        agent = self._agents[id]

        if region == CleanupRegion.APPLE:
            if apple_agent_map[agent.pos] != 0:
                return 0, 0
            apple_agent_map[agent.pos] = id
            waste_agent_map[agent.pos] = 0
            agent.region = CleanupRegion.APPLE
            if apple_map[agent.pos] != 0:
                apple_map[agent.pos] = 0
                return 1, 0
        else:
            if waste_agent_map[agent.pos] != 0:
                return 0, 0
            apple_agent_map[agent.pos] = 0
            waste_agent_map[agent.pos] = id
            agent.region = CleanupRegion.WASTE
            if waste_map[agent.pos] != 0:
                waste_map[agent.pos] = 0
                return 0, 1
        return 0, 0

    def move_agent(self, id, direction, apple_map, waste_map, apple_agent_map, waste_agent_map):
        """
        Move an agent.
        Returns a tuple (r, d) where r is the number of apples eaten and d is the number of dirt cleaned.
        """
        agent = self._agents[id]
        map = apple_map if agent.region == CleanupRegion.APPLE else waste_map
        agent_map = apple_agent_map if agent.region == CleanupRegion.APPLE else waste_agent_map
        new_pos = agent.pos + direction
        if new_pos < 0 or new_pos >= len(map):
            return 0, 0
        if agent_map[new_pos] != 0:
            return 0, 0

        agent_map[new_pos] = id
        agent_map[agent.pos] = 0
        agent.pos = new_pos

        if map[new_pos] != 0:
            map[new_pos] = 0
            if agent.region == CleanupRegion.APPLE:
                return 1, 0
            else:
                return 0, 1
        return 0, 0

    def compute_probabilities(self, num_dirt):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = num_dirt / self.potential_waste_area
        if waste_density >= self.thresholdDepletion:
            current_apple_spawn_prob = 0
            current_waste_spawn_prob = 0
        else:
            current_waste_spawn_prob = self.starting_waste_spawn_prob
            if waste_density <= self.thresholdRestoration:
                current_apple_spawn_prob = self.starting_apple_spawn_prob
            else:
                spawn_prob = (1 - (waste_density - self.thresholdRestoration)
                              / (self.thresholdDepletion - self.thresholdRestoration)) \
                             * self.starting_apple_spawn_prob
                current_apple_spawn_prob = spawn_prob
        return current_apple_spawn_prob, current_waste_spawn_prob

    def spawn_apples_and_waste(self, num_dirt, num_cleaners, current_apple_spawn_prob, current_waste_spawn_prob, apple_map, waste_map: np.ndarray, apple_agent_map, waste_agent_map):
        num_apples_spawned = 0
        num_waste_spawned = 0
        # spawn apples, multiple can spawn per step
        for x in range(self.potential_apple_area):
            rand_num = np.random.rand(1)[0]
            if rand_num < current_apple_spawn_prob and apple_agent_map[x] == 0 and apple_map[x] == 0:
                apple_map[x] = 1
                num_apples_spawned += 1

        # spawn one waste point, only one can spawn per step
        if num_dirt + num_cleaners < self.potential_waste_area:
            rand_num = np.random.rand(1)[0]
            if rand_num < current_waste_spawn_prob:
                remaining_locs = np.where(np.logical_and(waste_map == 0, waste_agent_map == 0))[0]
                if len(remaining_locs) > 0:
                    loc = random.choice(remaining_locs)
                    waste_map[loc] = 1
                    num_waste_spawned += 1

        return num_apples_spawned, num_waste_spawned

    def closest_objective(self, region, pos, apple_map=None, waste_map=None):
        """
        Returns a tuple (u, d) where u is the distance to the closest apple above the position and d is the distance to the closest apple below the position.
        """
        if apple_map is None:
            apple_map = self.apple_map
        if waste_map is None:
            waste_map = self.waste_map

        if region == CleanupRegion.APPLE:
            map = apple_map
        else:
            map = waste_map

        u = np.where(map[:pos] == 1)[0]
        u = np.inf if len(u) == 0 else pos - u[-1]
        d = np.where(map[pos+1:] == 1)[0]
        d = np.inf if len(d) == 0 else d[0] + 1
        return u, d

    def closest_agents(self, region, pos, apple_map=None, waste_map=None):
        """
        Returns a tuple (u, d) where u is the distance to the closest apple above the position and d is the distance to the closest apple below the position.
        """
        if apple_map is None:
            apple_map = self.apple_agent_map
        if waste_map is None:
            waste_map = self.waste_agent_map

        if region == CleanupRegion.APPLE:
            map = apple_map
        else:
            map = waste_map

        u = np.where(map[:pos] != 0)[0]
        u = np.inf if len(u) == 0 else pos - u[-1]
        d = np.where(map[pos+1:] != 0)[0]
        d = np.inf if len(d) == 0 else d[0] + 1
        return u, d

    def get_greedy_assignments(self, num_pickers: int, num_cleaners: int):
        """
        Returns a dictionary of greedy role assignments for each agent, where there are num_pickers pickers and num_cleaners cleaners.
        Assigns agents to the role that minimises their distance to the closest objective in that role.
        """
        assignments = {}
        agents = list(self._agents.values())
        apple_ordering = sorted(agents, key=lambda agent: min(self.closest_objective(CleanupRegion.APPLE, agent.pos)))
        waste_ordering = sorted(agents, key=lambda agent: min(self.closest_objective(CleanupRegion.WASTE, agent.pos)))
        for i in range(num_pickers):
            assignments[apple_ordering[i].agent_id] = CleanupRegion.APPLE
            waste_ordering.remove(apple_ordering[i]) # remove the agent from the waste ordering so that it can't be assigned to both roles
        for i in range(num_cleaners):
            assignments[waste_ordering[i].agent_id] = CleanupRegion.WASTE
        return assignments

    def get_greedy_actions(self, roles: dict[str, CleanupRegion]):
        """
        Returns a dictionary of greedy actions for each agent.
        """
        actions = {}
        for id in self.get_agent_ids():
            agent = self._agents[id]
            role = roles[id]
            closest_objective = self.closest_objective(agent.region, agent.pos)
            closest_agents = self.closest_agents(agent.region, agent.pos)
            if closest_objective[0] >= closest_objective[1]:
                direction = 1
            else:
                direction = -1

            actions[id] = (role, direction)

        return actions

    def simulate_step(self, actions: dict[str, tuple[CleanupRegion, int]]):
        """
        Simulate the future state of the environment after all agents perform their actions.
        Returns a tuple (observations, rewards) where observations is a dictionary of agent observations and rewards is a dictionary of agent rewards in the hypothetical future state.
        """
        # copy maps so that we can update it without affecting the originals to simulate the future state
        apple_map = self.apple_map.copy()
        waste_map = self.waste_map.copy()
        apple_agent_map = self.apple_agent_map.copy()
        waste_agent_map = self.waste_agent_map.copy()
        num_pickers = 0
        num_cleaners = 0

        rewards, num_apples, num_dirt, num_pickers, num_cleaners = self.perform_step(actions, apple_map, waste_map, apple_agent_map, waste_agent_map)

        observations = {
            'coordinator': (num_apples, num_dirt, num_pickers, num_cleaners),
        }
        for id in self.get_agent_ids():
            agent = self._agents[id]
            closest_objective = self.closest_objective(agent.region, agent.pos, apple_map=apple_map, waste_map=waste_map)
            closest_agents = self.closest_agents(agent.region, agent.pos, apple_map=apple_agent_map, waste_map=waste_agent_map)
            observations[id] = (closest_objective[0], closest_objective[1], closest_agents[0], closest_agents[1])

        return observations, rewards

    def render(self) -> None:
        """
        Render the environment.
        """
        apple_map = self.apple_map.copy()
        apple_map = np.expand_dims(apple_map, axis=0)
        apple_map = np.repeat(apple_map, 10, axis=0)
        plt.figure()
        plt.title("Apple Map")
        plt.imshow(apple_map)
        plt.show()

        waste_map = self.waste_map.copy()
        waste_map = np.expand_dims(waste_map, axis=0)
        waste_map = np.repeat(waste_map, 10, axis=0)
        plt.figure()
        plt.title("Waste Map")
        plt.imshow(waste_map)
        plt.show()
        apple_agent_map = self.apple_agent_map.copy()
        for id in self._agent_ids:
            apple_agent_map[np.where(apple_agent_map == id)] = 1
        apple_agent_map = np.expand_dims(apple_agent_map, axis=0)
        apple_agent_map = np.repeat(apple_agent_map, 10, axis=0)
        plt.figure()
        plt.title("Apple Agent Map")
        plt.imshow(apple_agent_map)
        plt.show()

        waste_agent_map = self.waste_agent_map.copy()
        for id in self._agent_ids:
            waste_agent_map[np.where(waste_agent_map == id)] = 1
        waste_agent_map = np.expand_dims(waste_agent_map, axis=0)
        waste_agent_map = np.repeat(waste_agent_map, 10, axis=0)
        plt.figure()
        plt.title("Waste Agent Map")
        plt.imshow(waste_agent_map)
        plt.show()
