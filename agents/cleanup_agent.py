import numpy as np

class CleanupAgent():
    def __init__(self, agent_id, start_pos):
        self.agent_id = agent_id
        self.pos = start_pos
        self.reward = 0

class GreedyCleanUpAgent(CleanupAgent):
    def __init__(self, agent_id, start_pos, region):
        super().__init__(agent_id, start_pos)
        self.region = region  # region == 1 for apples and -1 for waste
