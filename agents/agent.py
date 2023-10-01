import numpy as np

class Agent():
    def __init__(self, agent_id, start_pos, *args):
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.reward = 0

class ObjectiveAgent(Agent):
    def __init__(self, agent_id, start_pos, objective, *args):
        self.objective = objective
        super().__init__(agent_id, start_pos)