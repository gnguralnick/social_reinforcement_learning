import math
import numpy as np
from agents.agent import ObjectiveAgent
from environments.env import ObjectiveEnv
from typing import cast

class ContentMarketAgent(ObjectiveAgent):
    """
    An agent in a content market environment.
    The 'content' produced by this agent is its pursual of an objective.
    The agent has a probability of 'producing' each objective, which is updated at each timestep.
    The objective with the highest probability is the one that the agent will pursue, and is the agent's 'current objective'.
    """

    def __init__(self, agent_id, start_pos, env: ObjectiveEnv, preferred_objective, maximum_attention, objective_preferences, delay_sensitivity=1, *args):
        self.env = env

        objectives = self.env.objectives
        
        preferred_objective = preferred_objective if preferred_objective is not None else np.random.choice(list(self.objective_probs.keys()))
        self._objective_probs = {objective: 1 if objective == preferred_objective else 0 for objective in objectives}

        self._following_rates = {other_id: 0. for other_id in self.env.agents.keys() if other_id != agent_id}
        self._environmental_following_rate = 1 # agents start out only caring about the environmental reward

        self.maximum_attention = maximum_attention # sum of following rates and environmental following rate cannot exceed this value
        self.delay_sensitivity = delay_sensitivity

        self.other_agent_objective_preferences = objective_preferences # dictionary of the form {objective: preference} for each objective in the environment
                                                                       # indicates how much the agent likes other agents to pursue each objective
        super().__init__(agent_id, start_pos, preferred_objective)

    @property
    def following_rates(self) -> dict[str, float]:
        return self._following_rates
    
    @following_rates.setter
    def following_rates(self, following_rates: dict[str, float]):
        self._following_rates = following_rates
        self._environmental_following_rate = self.maximum_attention - sum(self._following_rates.values())

        if self._environmental_following_rate < 0:
            raise ValueError("Sum of following rates cannot exceed maximum attention")
    
    @property
    def objective_probs(self) -> dict[str, float]:
        return self._objective_probs
    
    @objective_probs.setter
    def objective_probs(self, objective_probs: dict[str, float]):
        self._objective_probs = objective_probs
        self.objective = max(self._objective_probs, key=self._objective_probs.get)
        
    def consumption_utility(self, last_objectives: dict[str, str], last_reward: float) -> float:
        """
        Returns the agent's utility from observing the other agents' objectives and rewards according to its following rates.
        """
        utility = 0
        for other_id in self.env.agents.keys():
            if other_id == self.agent_id:
                continue
            if self.following_rates[other_id] == 0:
                continue
            other = cast(ContentMarketAgent, self.env.agents[other_id])
            
            other_last_objective = last_objectives[other_id]
            other_last_objective_prob = other.objective_probs[other_last_objective]
            other_last_objective_preference = self.other_agent_objective_preferences[other_last_objective]

            delay_mult = math.exp(-self.delay_sensitivity / self.following_rates[other_id])

            utility += other_last_objective_prob * other_last_objective_preference * delay_mult
        utility += self._environmental_following_rate * last_reward
        return utility

    @staticmethod
    def from_objective_agent(agent: ObjectiveAgent, env: ObjectiveEnv):
        return ContentMarketAgent(agent.agent_id, agent.pos, env, agent.objective)