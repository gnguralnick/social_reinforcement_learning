import numpy as np

from environments.cleanup_env import CleanupEnv
from models.model import Model
import typing


class CleanupHeuristicModel(Model):

    def __init__(self, env, num_outputs, model_config, name):
        super(CleanupHeuristicModel, self).__init__(env, num_outputs, model_config, name)
        if not isinstance(env, CleanupEnv):
            raise ValueError("CleanupHeuristicModel only works with CleanupEnv")
        self.env = typing.cast(CleanupEnv, env)

    def reassign_regions_of_greedy_agents(self):
        agent_frequency_in_dirt = self.env.num_dirt / (self.env.num_apples + self.env.num_dirt)
        num_agents_to_be_assigned_to_dirt = round(self.env.num_agents * agent_frequency_in_dirt)
        agents_assigned_to_dirt = [agent for agent in self.env.agents.values() if agent.region == -1]
        agents_assigned_to_apples = [agent for agent in self.env.agents.values() if agent.region == 1]
        if len(agents_assigned_to_dirt) < num_agents_to_be_assigned_to_dirt:
            agents_assigned_to_apples.sort(key=lambda agent: self.find_nearest_waste_from_agent(agent)[1])
            for i in range(num_agents_to_be_assigned_to_dirt - len(agents_assigned_to_dirt)):
                agents_assigned_to_apples[i].region = -1
        elif len(agents_assigned_to_dirt) > num_agents_to_be_assigned_to_dirt:
            agents_assigned_to_dirt.sort(key=lambda agent: self.find_nearest_apple_from_agent(agent)[1])
            for i in range(len(agents_assigned_to_dirt) - num_agents_to_be_assigned_to_dirt):
                agents_assigned_to_dirt[i].region = 1

    def greedily_move_to_closest_object(self):
        """
        Each agent moves to the closest object
        """
        actions = {}
        for agent in self.env.agents.values():
            actions[agent.agent_id] = self.get_greedy_action(agent)
        return actions

    def find_nearest_apple_from_agent(self, agent):
        x, y = agent.pos
        closest_x, closest_y, min_distance = -1, -1, float('inf')
        for i in range(self.env.height):
            for j in range(self.env.width):
                if self.env.map[i][j] == 1 and abs(i - x) + abs(j - y) <= min_distance:
                    min_distance = abs(i - x) + abs(j - y)
                    closest_x, closest_y = i, j
        return [closest_x, closest_y], min_distance

    def find_nearest_waste_from_agent(self, agent):
        x, y = agent.pos
        closest_x, closest_y, min_distance = -1, -1, float('inf')
        for i in range(self.env.height):
            for j in range(self.env.width):
                if self.env.map[i][j] == -1 and abs(i - x) + abs(j - y) <= min_distance:
                    min_distance = abs(i - x) + abs(j - y)
                    closest_x, closest_y = i, j
        return [closest_x, closest_y], min_distance

    def get_greedy_action(self, agent):
        if agent.region == 1:
            nearest_obj = self.find_nearest_apple_from_agent(agent)[0]
        else:
            nearest_obj = self.find_nearest_waste_from_agent(agent)[0]
        if agent.pos[0] == nearest_obj[0]:
            if nearest_obj[1] < agent.pos[1]:
                return 3
            return 1
        if agent.pos[0] > nearest_obj[0]:
            return 0
        return 2

    def predict(self, obs, **kwargs):
        self.reassign_regions_of_greedy_agents()
        actions = np.zeros((1, self.num_outputs, self.num_actions))
        for agent in self.env.agents.values():
            action = self.get_greedy_action(agent)
            actions[0][int(agent.agent_id)][action] = 1
        return actions
