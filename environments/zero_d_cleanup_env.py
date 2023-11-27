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
        self.total_reward_by_agent = {id: 0 for id in self.get_agent_ids()}
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
            "total_reward_by_agent": self.total_reward_by_agent,
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
        for id in self.get_agent_ids():
            self.total_reward_by_agent[id] += step_reward[id]
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
            "total_reward_by_agent": self.total_reward_by_agent,
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
        apple_agents = list(np.random.permutation(list(apple_agent_ids)))
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
    
    def transition_P(self, s0, s1):
        delta_a = s0[0] - s1[0]
        delta_d = s0[1] - s1[1]
        p1 = float(comb(s0[0], delta_a)*comb(self.area-s0[0], s1[2]-delta_a)) / comb(self.area, s1[2])
        p2 = float(comb(s0[1], delta_d)*comb(self.area-s0[1], s1[3]-delta_d)) / comb(self.area, s1[3])
        p = p1 * p2
        return p
    
    def simulate_future_state(self, new_p, new_c):
        s_original = np.array([self.num_apples, self.num_dirt])

        d = []
        tp = []
        for p in range(new_p+1):
            for c in range(new_c+1):
                if (self.num_apples-p) < 0 or (self.num_agents-c) < 0:
                    continue
                s_new = np.array([self.num_apples-p, self.num_dirt-c, new_p, new_c])
                s_new_input = np.array([float(self.num_apples-p), float(self.num_dirt-c)])
                transition_prob = self.transition_P(s_original, s_new)
                dirt_density = (self.num_dirt-c) / self.area
                if dirt_density >= self.thresholdDepletion:  # nothing will grow
                    u_input0 = torch.tensor(s_new_input).float().unsqueeze(0)
                    d.append(u_input0)
                    tp.append(transition_prob)
                else:
                    apple_prob = (1 - (dirt_density - self.thresholdRestoration)/(self.thresholdDepletion - self.thresholdRestoration)) * self.starting_apple_spawn_prob
                    dirt_prob = 0.5
                    apple_potential = self.area - (self.num_apples-p)
                    # for apple_g in range(10):  # estimate, for performance, large apple_g will have extremely small prob.
                    #     a_p = comb(apple_potential, apple_g) * (apple_prob**apple_g) * ((1-apple_prob)**(apple_potential-apple_g))
                    #     s_new_input[0] += apple_g
                    #     u_input0 = torch.tensor(s_new_input).float().unsqueeze(0).to(device)
                    #     d.append(u_input0)
                    #     tp.append(transition_prob * a_p * dirt_prob)
                    #     s_new_input[1] += 1
                    #     u_input0 = torch.tensor(s_new_input).float().unsqueeze(0).to(device)
                    #     d.append(u_input0)
                    #     tp.append(transition_prob * a_p * dirt_prob)
                    s_new_input[0] += apple_prob * apple_potential
                    s_new_input[1] += dirt_prob
                    u_input0 = torch.tensor(s_new_input).float().unsqueeze(0)
                    d.append(u_input0)
                    tp.append(transition_prob)
        return d, tp
    
    def get_immediate_reward(self, n_pickers):
        return (self.num_apples * n_pickers) / self.area