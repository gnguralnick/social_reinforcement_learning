import numpy as np


class CleanupAgent():
    def __init__(self, agent_id, start_pos):
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.holding = "0"  # what is the agent holding
        self.reward = 0
