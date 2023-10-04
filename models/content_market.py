import numpy as np
from agents.content_market_agent import ContentMarketAgent
from models.model import ObjectiveModel
from environments.env import ObjectiveEnv

class BasicContentMarketModel(ObjectiveModel):
    """
    The 'perfect information' model for the content market environment.
    Agents will have access to each other's preferences and try to optimize towards market equilibrium in terms of following rates and production rates.
    """

    def __init__(self, env: ObjectiveEnv, num_outputs, model_config, name):
        super().__init__(env, num_outputs, model_config, name)
        self.agents_dict = {agent_id: ContentMarketAgent.from_objective_agent(agent, self.env) for agent_id, agent in self.env.agents.items()}


    def reassign_agent_objectives(self):
        """
        Update agent production and following rates to optimize towards market equilibrium.
        """
        agents = self.agents_dict.values()
        last_rewards = self.env.last_rewards
        # TODO: update agent following rates
        # TODO: update agent production rates (objective probabilities)
        return super().reassign_agent_objectives()
    
    def predict(self, obs, **kwargs):
        """
        Sample agent actions from agent production rates
        i.e. for each agent, choose objective with probability equal to agent's production rate
        and then greedily move agent towards chosen objective
        """
        self.reassign_agent_objectives()
        actions = np.zeros((1, self.num_outputs, self.num_actions))
        for agent_id in self.agents_dict:
            agent = self.agents_dict[agent_id]
            cumulative_objective_probs = np.cumsum(list(agent.objective_probs.values()))
            objective_ind = np.searchsorted(cumulative_objective_probs, np.random.rand())
            objective = list(agent.objective_probs.keys())[objective_ind]

            action = self.env.get_greedy_action(agent, objective)[0]
            actions[0][int(agent_id)][action] = 1
        return actions
