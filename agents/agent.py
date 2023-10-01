import numpy as np

class Agent():
    def __init__(self, agent_id, start_pos=None, *args):
        self.agent_id = agent_id
        self.pos = np.array(start_pos) if start_pos is not None else None
        self.reward = 0
    
    def set_pos(self, pos):
        self.pos = np.array(pos)

class ObjectiveAgent(Agent):
    def __init__(self, agent_id, start_pos, objective, *args):
        self.objective = objective
        super().__init__(agent_id, start_pos)