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

    def __init__(self, agent_id, start_pos, env: ObjectiveEnv, curr_objective, *args):
        self.env = env
        objectives = self.env.objectives
        self.objective_probs = {objective: 1 / len(objectives) for objective in objectives}
        curr_objective = curr_objective if curr_objective is not None else np.random.choice(list(self.objective_probs.keys()), p=list(self.objective_probs.values()))
        super().__init__(agent_id, start_pos, curr_objective)

    @staticmethod
    def from_objective_agent(agent: ObjectiveAgent, env: ObjectiveEnv):
        return ContentMarketAgent(agent.agent_id, agent.pos, env, agent.objective)