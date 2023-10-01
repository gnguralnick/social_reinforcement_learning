from abc import abstractmethod
from ray.rllib.env import MultiAgentEnv

from agents.agent import ObjectiveAgent, Agent

class AgentEnv(MultiAgentEnv):

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents
        super().__init__()

class ObjectiveEnv(AgentEnv):
    """
    A MultiAgentEnv where agents must choose to persue one of a number of objectives.
    Not all of these objectives provide a direct reward, however some objectives
    are required to be completed for reward-generating objectives to be accessible.
    """

    def __init__(self, agents: dict[str, ObjectiveAgent], objectives: list):
        self.objectives = objectives
        super().__init__(agents)
    
    @abstractmethod
    def find_nearest_objective(self, agent: ObjectiveAgent):
        """
        Finds the nearest objective to the given agent, based on the agent's position and current objective.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_greedy_action(self, agent: ObjectiveAgent):
        """
        Returns the action that will move the agent towards its current objective.
        """
        raise NotImplementedError

