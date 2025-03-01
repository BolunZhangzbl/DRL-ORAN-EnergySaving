"""
constants.py

This file contains all global constants used in the ORAN environment and DRL agent configuration.
These constants are derived from the `config.yaml` file and should not be modified during runtime.
"""

# ======================
# ORAN Environment Configuration
# ======================

# Center frequency of the ORAN system (in Hz)
CENTER_FREQ = 850e6

# Bandwidth of the ORAN system (in Hz)
BANDWIDTH = 20e6

# Bandwidth per resource block (RB) (in Hz)
BANDWIDTH_PER_RB = 2e5

# PRB Efficiency (in Mbps)
RB_EFFICIENCY = 1e6

# Number of gNBs in the ORAN system
NUM_GNB = 7

# Number of resource blocks (RBs) available
NUM_RBS = 100

# Inter-distance between gNBs (in meters)
INTER_DISTANCE_GNB = 1700

# Number of UEs per gNB
NUM_UES_PER_GNB = 9

# Maximum transmit power of gNBs (in dBm)
POWER_MAX = 15  # dBm

# Transmission Time Interval (TTI) (in seconds)
TTI = 1e-3

# Noise power density (in W/Hz)
NOISE_POWER_DENSITY = 3.98e-21

# Std for shadowing (in dB)
SHADOWING_STD = 8

# Time Step (in Sec)
TIME_STEP = 0.1

# ======================
# DRL Agent Configuration
# ======================

# Maximum number of steps per episode
MAX_STEP = 50

# Number of episodes for training
NUM_EPISODES = 500

# Number of last episodes to consider for evaluation
LAST_N = 10

# Learning rate for the actor network
ACTOR_LR = 3e-4

# Learning rate for the critic network
CRITIC_LR = 1e-3

# Learning rate for the DQN network
DQN_LR = 1e-3

# Discount factor for future rewards
GAMMA = 0.99

# Initial exploration rate (epsilon-greedy strategy)
EPSILON = 1.0

# Minimum exploration rate
EPSILON_MIN = 0.01

# Decay rate for exploration rate
EPSILON_DECAY = 0.99

# Size of the state space
STATE_SPACE = 85

# Size of the action space
ACTION_SPACE = 2**7