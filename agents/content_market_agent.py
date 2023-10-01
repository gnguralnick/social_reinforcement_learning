import numpy as np
from agents.agent import ObjectiveAgent
from environments.env import ObjectiveEnv

class ContentMarketAgent(ObjectiveAgent):
    """
    An agent in a content market environment.
    The 'content' produced by this agent is its pursual of an objective.
    The agent has a probability of 'producing' each objective, which is updated at each timestep.
    The objective with the highest probability is the one that the agent will pursue, and is the agent's 'current objective'.
    """

    def __init__(self, agent_id, start_pos, env: ObjectiveEnv, curr_objective, maximum_attention, *args):
        self.env = env

        objectives = self.env.objectives
        self.objective_probs = {objective: 1 / len(objectives) for objective in objectives}
        curr_objective = curr_objective if curr_objective is not None else np.random.choice(list(self.objective_probs.keys()), p=list(self.objective_probs.values()))

        self._following_rates = {other_id: 0. for other_id in self.env.agents.keys() if other_id != agent_id}
        self._environmental_following_rate = 1 # agents start out only caring about the environmental reward

        self.maximum_attention = maximum_attention # sum of following rates and environmental following rate cannot exceed this value
        super().__init__(agent_id, start_pos, curr_objective)

    @property
    def following_rates(self) -> dict[str, float]:
        return self._following_rates
    
    @following_rates.setter
    def following_rates(self, following_rates: dict[str, float]):
        self._following_rates = following_rates
        self._environmental_following_rate = self.maximum_attention - sum(self._following_rates.values())

        if self._environmental_following_rate < 0:
            raise ValueError("Sum of following rates cannot exceed maximum attention")

    @staticmethod
    def from_objective_agent(agent: ObjectiveAgent, env: ObjectiveEnv):
        return ContentMarketAgent(agent.agent_id, agent.pos, env, agent.objective)