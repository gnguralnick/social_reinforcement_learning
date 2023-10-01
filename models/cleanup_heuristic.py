from environments.cleanup_env import CleanupEnv
from models.model import ObjectiveModel


class CleanupHeuristicModel(ObjectiveModel):

    def __init__(self, env: CleanupEnv, num_outputs, model_config, name):
        super(CleanupHeuristicModel, self).__init__(env, num_outputs, model_config, name)
        self.env = env

    def reassign_agent_objectives(self):
        agent_frequency_in_waste = self.env.num_waste / (self.env.num_apples + self.env.num_waste)
        num_agents_to_be_assigned_to_waste = round(self.env.num_agents * agent_frequency_in_waste)
        agents_assigned_to_waste = [agent for agent in self.env.agents.values() if agent.objective == 'waste']
        agents_assigned_to_apples = [agent for agent in self.env.agents.values() if agent.objective =='apples']
        if len(agents_assigned_to_waste) < num_agents_to_be_assigned_to_waste:
            agents_assigned_to_apples.sort(key=lambda agent: self.env.find_nearest_waste_from_agent(agent)[1])
            for i in range(num_agents_to_be_assigned_to_waste - len(agents_assigned_to_waste)):
                agents_assigned_to_apples[i].region = -1
        elif len(agents_assigned_to_waste) > num_agents_to_be_assigned_to_waste:
            agents_assigned_to_waste.sort(key=lambda agent: self.env.find_nearest_apple_from_agent(agent)[1])
            for i in range(len(agents_assigned_to_waste) - num_agents_to_be_assigned_to_waste):
                agents_assigned_to_waste[i].region = 1