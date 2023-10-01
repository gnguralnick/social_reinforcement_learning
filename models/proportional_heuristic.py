from models.model import ObjectiveModel

class ProportionalHeuristicModel(ObjectiveModel):
    """
    Assigns agents to objectives proportional to the quantity of each objective.
    Greedily moves agents towards their assigned objective.
    """

    def reassign_agent_objectives(self):
        """
        Reassigns objectives to agents.
        """
        total_objective_quantity = sum(self.env.objectives.values())

        objective_stats = {}

        unassigned_agents: list[str] = []

        for objective in self.env.objectives:
            proportion = round(self.env.objectives[objective] / total_objective_quantity)
            desired_num_agents = self.num_outputs * proportion
            current_assignments = [agent for agent in self.env.agents if self.env.agents[agent].objective == objective]
            current_assignments.sort(key=lambda agent: self.env.find_nearest_objective(agent)[1], reverse=True)

            if len(current_assignments) == desired_num_agents:
                continue

            objective_stats[objective] = {
                'desired_num_agents': desired_num_agents,
                'current_assignments': current_assignments
            }

            if len(current_assignments) > desired_num_agents:
                for i in range(len(current_assignments) - desired_num_agents):
                    current_assignments[i].objective = None
                    unassigned_agents.append(current_assignments[i])

        for objective, stats in objective_stats.items():
            unassigned_agents.sort(key=lambda agent: self.env.find_nearest_objective(agent)[1])

            for i in range(stats['desired_num_agents'] - len(stats['current_assignments'])):
                self.env.agents[unassigned_agents[i]].objective = objective