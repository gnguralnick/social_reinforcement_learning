from abc import ABCMeta, abstractmethod
from ray.rllib.env import MultiAgentEnv

from agents.agent import ObjectiveAgent, Agent

class AgentEnv(MultiAgentEnv):

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents
        super().__init__()

class ObjectiveEnv(AgentEnv, metaclass=ABCMeta):
    """
    A MultiAgentEnv where agents must choose to persue one of a number of objectives.
    Not all of these objectives provide a direct reward, however some objectives
    are required to be completed for reward-generating objectives to be accessible.

    self.objectives is a dictionary of the form {objective_name: objective_quantity}.
    """

    def __init__(self, agents: dict[str, ObjectiveAgent], objectives: dict[str, int]):
        self.objectives = objectives
        super().__init__(agents)
        self.agents = agents
        self.last_rewards = {agent_id: 0 for agent_id in self.agents}
    
    @abstractmethod
    def find_nearest_objective(self, agent: ObjectiveAgent, objective: str = None):
        """
        Finds the nearest objective to the given agent, based on the agent's position.
        If objective is None, the agent's current objective is used.
        """
        raise NotImplementedError
    
    def get_greedy_action(self, agent: ObjectiveAgent, objective: str = None):
        nearest_obj = self.find_nearest_objective(agent, objective)[0]
        if agent.pos[0] == nearest_obj[0]:
            if nearest_obj[1] < agent.pos[1]:
                return 3
            return 1
        if agent.pos[0] > nearest_obj[0]:
            return 0
        return 2
    
    def get_objective_quantity(self, objective: str):
        """
        Returns the quantity of the given objective.
        """
        return self.objectives[objective]