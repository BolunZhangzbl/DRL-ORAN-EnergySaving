# -- Public Imports
import argparse

# -- Private Imports
from constants import *

# -- Global Variables


# -- Functions

def get_config():
    parser = argparse.ArgumentParser(description="DRL Agent Configuration")

    # Add arguments for each configuration parameter
    parser.add_argument('--max_step', type=int, default=100, help='Maximum number of steps per episode')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes for training')
    parser.add_argument('--last_n', type=int, default=10, help='Number of last episodes to consider for evaluation')
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='Learning rate for the actor network')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Learning rate for the critic network')
    parser.add_argument('--dqn_lr', type=float, default=1e-3, help='Learning rate for the DQN network')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate (epsilon-greedy strategy)')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.9999, help='Decay rate for exploration rate')
    parser.add_argument('--state_space', type=int, default=50, help='Size of the state space')
    parser.add_argument('--action_space', type=int, default=2 ** NUM_GNB, help='Size of the action space')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Add environment and agent arguments with restricted choices
    parser.add_argument('--env', type=str, default='sim', choices=['sim', 'testbed'],
                        help='Environment to use (sim or testbed)')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn'],
                        help='Agent to use (currently only DQN is supported)')
    # Parse the arguments
    args = parser.parse_args()

    return args