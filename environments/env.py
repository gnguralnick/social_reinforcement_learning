from abc import abstractmethod
from ray.rllib.env import MultiAgentEnv

from agents.agent import ObjectiveAgent, Agent
from agents.content_market_agent import ContentMarketAgent

class AgentEnv(MultiAgentEnv):

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents
        super().__init__()

class ObjectiveEnv(AgentEnv):
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
    
    @abstractmethod
    def find_nearest_objective(self, agent: ObjectiveAgent, objective: str = None):
        """
        Finds the nearest objective to the given agent, based on the agent's position.
        If objective is None, the agent's current objective is used.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_greedy_action(self, agent: ObjectiveAgent, objective: str = None):
        """
        Returns the action that will move the agent towards an objective.
        If objective is None, the agent's current objective is used.
        """
        raise NotImplementedError
    
    def get_objective_quantity(self, objective: str):
        """
        Returns the quantity of the given objective.
        """
        return self.objectives[objective]


def convert_to_content_market(env: ObjectiveEnv):
    """
    Converts an ObjectiveEnv to a ContentMarketEnv by replacing all agents with ContentMarketAgents.
    """
    env.agents = {agent_id: ContentMarketAgent(agent_id, env.agents[agent_id].pos, env, env.agents[agent_id].objective) for agent_id in env.agents}
    return env