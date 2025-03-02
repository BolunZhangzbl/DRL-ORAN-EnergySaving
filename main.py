# -- Public Imports
import argparse

# -- Private Imports
from train import run_drl
from config import get_config

# -- Global Variables


# -- Functions

if __name__ == '__main__':
    # Set up argument parser
    args = get_config()
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    run_drl(args)