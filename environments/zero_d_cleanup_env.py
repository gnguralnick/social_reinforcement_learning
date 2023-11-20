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

    def __init__(self, agent_ids, num_agents=10, area=150, thresholdDepletion: float=0.4, thresholdRestoration: float=0, wasteSpawnProbability: float=0.5, appleRespawnProbability: float=0.05, dirt_multiplier=10):
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

        #self.agents: dict[str, GreedyCleanUpAgent] = {}
        #self.setup_simple_agent()

        self.num_dirt = 78
        self.num_apples = 0
        #self.dirt_agent = self.num_agents  # init all dirt cleaner
        #self.apple_agent = 0
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

        super().__init__()

    # def setup_simple_agent(self):
    #     for i in range(self.num_agents):
    #         self.agents[str(i)] = GreedyCleanUpAgent(str(i), [0, 0], -1)

    def reset(self, seed: int | None = None, options: dict = {}) -> tuple:
        """
        Reset the environment.
        """
        # Set seed
        super().reset(seed=seed)
        self.timestamp = 0
        #self.agents = {}
        #self.setup_simple_agent()

        self.num_dirt = 78
        self.num_apples = 0
        #self.dirt_agent = self.num_agents
        #self.apple_agent = 0
        self.current_apple_spawn_prob = self.starting_apple_spawn_prob
        self.current_waste_spawn_prob = self.starting_waste_spawn_prob
        self.compute_probabilities()

        self.total_apple_consumed = 0
        self.step_apple_consumed = 0

        observations = {
            # id: {
            #     "apple": np.array([self.num_apples]),
            #     "dirt": np.array([self.num_dirt]),
            #     "picker": np.array([0]),
            #     "cleaner": np.array([0])
            # } for id in self.get_agent_ids()
            id: np.array([self.num_apples, self.num_dirt, 0, 0]) for id in self.get_agent_ids()
        }

        info = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": np.array([self.num_apples]),
            "dirt": np.array([self.num_dirt]),
            "picker": np.array([0]),
            "cleaner": np.array([0])
        }

        return observations, info

    def step(self, action_dict: dict) -> tuple:
        """
        Take a step in the environment.
        """
        # num_apple_agents = 0
        # num_dirt_agents = 0
        # for agent_id in self.get_agent_ids():
        #     if action_dict[agent_id] == 0:
        #         num_apple_agents += 1
        #     else:
        #         num_dirt_agents += 1

        apple_agents = [agent_id for agent_id in self.get_agent_ids() if action_dict[agent_id] == 0]
        dirt_agents = [agent_id for agent_id in self.get_agent_ids() if action_dict[agent_id] == 1]

        observations = {}
        rewards = {}
        dones = {}
        
        self.timestamp += 1
        self.step_apple_consumed = 0
        # if self.heuristic:
        #     agent_frequency_in_dirt = self.num_dirt / (self.num_apples + self.num_dirt)
        #     num_agents_to_be_assigned_to_dirt = round(self.num_agents * agent_frequency_in_dirt)
        #     agents_assigned_to_dirt = [agent for agent in self.agents.values() if agent.region == -1]
        #     agents_assigned_to_apples = [agent for agent in self.agents.values() if agent.region == 1]
        #     if len(agents_assigned_to_dirt) < num_agents_to_be_assigned_to_dirt:
        #         for i in range(num_agents_to_be_assigned_to_dirt - len(agents_assigned_to_dirt)):
        #             agents_assigned_to_apples[i].region = -1
        #             self.apple_agent -= 1
        #             self.dirt_agent += 1
        #     elif len(agents_assigned_to_dirt) > num_agents_to_be_assigned_to_dirt:
        #         for i in range(len(agents_assigned_to_dirt) - num_agents_to_be_assigned_to_dirt):
        #             agents_assigned_to_dirt[i].region = 1
        #             self.dirt_agent -= 1
        #             self.apple_agent += 1
        # else:  # use U-network to generate roles
        #     d = {}
        #     for agent in [self.agents[key] for key in sorted(self.agents)]:
        #         inf = np.array([self.num_apples, self.num_dirt, self.apple_agent, self.dirt_agent])
        #         if (self.num_apples, self.num_dirt, self.apple_agent, self.dirt_agent) in d:
        #             u_t = d[(self.num_apples, self.num_dirt, self.apple_agent, self.dirt_agent)]
        #         else:
        #             u_input0 = torch.tensor(inf).float().unsqueeze(0).to(device)
        #             u_t = centralAgent.u_network(u_input0)  # current future est.
        #             d[(self.num_apples, self.num_dirt, self.apple_agent, self.dirt_agent)] = u_t

        #         # What if I deflect?
        #         if agent.region == 1:
        #             self.apple_reward = u_t.item()
        #             inf[2] -= 1
        #             inf[3] += 1
        #         elif agent.region == -1:
        #             self.dirt_reward = u_t.item()
        #             inf[2] += 1
        #             inf[3] -= 1
        #         new_tup = (inf[0], inf[1], inf[2], inf[3])
        #         if new_tup in d:
        #             u_tp = d[new_tup]
        #         else:
        #             u_input1 = torch.tensor(inf).float().unsqueeze(0).to(device)
        #             u_tp = centralAgent.u_network(u_input1)
        #             d[new_tup] = u_tp

        #         if agent.region == 1:
        #             self.dirt_reward = u_tp.item()
        #             self.apple_agent -= 1
        #         else:
        #             self.apple_reward = u_tp.item()
        #             self.dirt_agent -= 1

        #         # make decision
        #         if random.random() > max(self.epsilon, 0.2):
        #             if self.dirt_reward >= self.apple_reward:
        #                 agent.region = -1
        #                 self.dirt_agent += 1
        #             else:
        #                 agent.region = 1
        #                 self.apple_agent += 1
        #         else:
        #             choice = np.random.choice(2)
        #             if choice == 0:
        #                 agent.region = 1
        #                 self.apple_agent += 1
        #             else:
        #                 agent.region = -1
        #                 self.dirt_agent += 1
        # print("=======")
        # print(f"INFO: Apple Number: {self.num_apples}, Dirt Number: {self.num_dirt}")
        # print(f"INFO: Num Apple Agents: {len(apple_agents)}, Num Dirt Agents: {len(dirt_agents)}")
        #interim_input = np.array([self.num_apples, self.num_dirt, num_apple_agents, num_dirt_agents])

        step_reward = self.step_reward_calculation(apple_agents)
        total_reward = sum(step_reward.values())
        self.step_apple_consumed = total_reward
        self.total_apple_consumed += total_reward
        self.step_dirt_calculation(dirt_agents)
        # self.step_apple_consumed += self.step_dirt_calculation()*0
        # print(f"INFO: Step Reward: {step_reward}")
        self.compute_probabilities()
        new_apple, new_dirt = self.spawn_apples_and_waste()
        # print(f"INFO: New Apple: {new_apple}, New Dirt: {new_dirt}")
        # print("=======")

        observations = {
            # id: {
            #     "apple": np.array([self.num_apples]),
            #     "dirt": np.array([self.num_dirt]),
            #     "num_picker": np.array(len(apple_agents)),
            #     "num_cleaner": np.array(len(dirt_agents))
            # } for id in self.get_agent_ids()
            id: np.array([self.num_apples, self.num_dirt, len(apple_agents), len(dirt_agents)]) for id in self.get_agent_ids()
        }
        #self.epsilon = self.epsilon * self.epsilon_decay

        # rewards["apple"] = self.total_apple_consumed
        # rewards["step_apple"] = self.step_apple_consumed
        rewards = step_reward
        dones["__all__"] = self.timestamp == 1000

        infos = {
            'total_apple_consumed': self.total_apple_consumed,
            'step_apple_consumed': self.step_apple_consumed,
            "apple": np.array([self.num_apples]),
            "dirt": np.array([self.num_dirt]),
            "picker": np.array([0]),
            "cleaner": np.array([0])
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

    # def generate_info(self, num_apple_agents, num_dirt_agents):
    #     return {"apple": self.num_apples, "dirt": self.num_dirt, "x1": 0,
    #             "x2": 0, "x3": 0, "picker": num_apple_agents, "cleaner": num_dirt_agents}

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
        for _ in range(self.area):
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