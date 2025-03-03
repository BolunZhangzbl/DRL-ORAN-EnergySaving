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
RB_EFFICIENCY = 2e6

# Number of gNBs in the ORAN system
NUM_GNB = 7

# Number of resource blocks (RBs) available
NUM_RBS = 100

# Inter-distance between gNBs (in meters)
INTER_DISTANCE_GNB = 1700

# Number of UEs per gNB
NUM_UES_PER_GNB = 9

# Maximum transmit power of gNBs (in W)
POWER_TX = 40

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

# Reward threshold
REWARD_THRESHOLD = 1e8