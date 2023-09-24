import numpy as np


class CleanupAgent():
    def __init__(self, agent_id, start_pos):
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.holding = "0"  # what is the agent holding
        self.reward = 0

class GreedyCleanUpAgent(CleanupAgent):
    def __init__(self, agent_id, start_pos, region):
        super().__init__(agent_id, start_pos)
        self.region = region  # region == 1 for apples and -1 for waste
