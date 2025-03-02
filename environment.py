# -- Public Imports


# -- Private Imports
from oran import ORAN
from utils import *
from constants import *

# -- Global Variables


# -- Functions

class ORANSimEnv:
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

        if reward >= REWARD_THRESHOLD:
            self.done = True

        self.current_time += TIME_STEP

        return next_state, reward, self.done

    def reset(self):
        self.oran = ORAN()
        self.current_time = 0

        return self.oran.get_state_oran()


class ORANTestbedEnv:
    """
    Environment class for interacting with an external hardware testbed.
    """
    def __init__(self, testbed_config):
        """
        Initialize the testbed environment.

        Args:
            testbed_config (dict): Configuration for connecting to the testbed.
        """
        self.testbed_config = testbed_config
        self.done = False
        self.current_time = 0
        self.simulation_time = 10  # 10 sec

        # Initialize connection to the testbed (placeholder)
        self.connect_to_testbed()

    def connect_to_testbed(self):
        """
        Establish a connection to the external hardware testbed.
        """
        # Placeholder for testbed connection logic
        print("Connecting to the testbed...")
        # Example: self.testbed = TestbedAPI(self.testbed_config)

    def get_observation(self):

        # Placeholder for retrieving observation from the testbed
        print("Retrieving observation from the testbed...")
        # Example: return self.testbed.get_state()
        return np.random.random((50,))  # Placeholder state

    def execute_action(self, action):

        # Placeholder for executing action on the testbed
        print(f"Executing action {action} on the testbed...")
        # e.g., self.testbed.execute(action)
    #           return self.testbed.get_state()
        return np.random.random((50,))  # Placeholder next state

    def calculate_reward(self, state):

        # Placeholder for reward calculation logic
        reward = np.sum(state)  # Example reward calculation
        return reward

    def step(self, action):

        next_state = self.execute_action(action)
        reward = self.calculate_reward(next_state)
        self.current_time += 1

        if reward >= REWARD_THRESHOLD:
            self.done = True

        return next_state, reward, self.done

    def reset(self):

        print("Resetting the testbed environment...")
        self.done = False
        self.current_time = 0
        return self.get_observation()


class UnifiedEnv:
    def __init__(self, env_type, config=None):

        self.env_type = env_type
        self.config = config

        if self.env_type == 'sim':
            self.env = ORANSimEnv()
        elif self.env_type == 'testbed':
            self.env = ORANTestbedEnv()
        else:
            raise ValueError("Invalid environment type. Choose 'sim' or 'testbed'.")

    def step(self, action):

        return self.env.step(action)

    def reset(self):

        return self.env.reset()