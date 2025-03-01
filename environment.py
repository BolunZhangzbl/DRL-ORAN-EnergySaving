# -- Public Imports


# -- Private Imports
from oran import ORAN
from utils import *
from constants import *

# -- Global Variables


# -- Functions

class ORANEnv:
    """
    Environment class for DRL training based on the ORAN system.
    """

    def __init__(self):
        """
        Initialize the ORAN environment.
        """
        self.oran = ORAN()
        self.current_time = 0
        self.simulation_time = 10  # 10 sec

        self.done = False

    def step(self, action):

        self.oran.update_system(self.current_time, action)

        next_state = self.oran.get_state_oran()

        reward = self.oran.get_sum_reward()

        if reward >= 10:
            self.done = True

        return next_state, reward, self.done

    def reset(self):
        self.oran = ORAN()
        self.current_time = 0

        return self.oran.get_state_oran()
