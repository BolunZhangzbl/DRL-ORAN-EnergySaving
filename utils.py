# -- Public Imports
import numpy as np

# -- Private Imports
from constants import *

# -- Global Variables


# -- Functions


class ActionMapper:
    def __init__(self, minVal, maxVal):
        # Total number of discrete actions
        self.minVal = minVal
        self.maxVal = maxVal
        self.num_actions = (maxVal - minVal + 1)
        self.actions = list(range(minVal, maxVal+1))

    def idx_to_action(self, idx):
        """
        Map an index to a unique action
        """
        if idx < 0 or idx >= self.num_actions:
            raise ValueError(f"Index {idx} out of valid range [0, {self.num_actions - 1}]")

        return int(self.actions[idx])

    def idx_to_bool_action(self, idx):
        if idx < self.minVal or idx > self.maxVal:
            raise ValueError(f"Action index {idx} is out of range [{self.minVal}, {self.maxVal}]")

            # Convert the integer index to a binary string
        binary_str = format(idx, f"0{self.num_gnbs}b")

        # Convert the binary string to a list of boolean values
        return [bool(int(bit)) for bit in binary_str]


def convert_dict_to_globals(d):
    """
    Load the config.yaml file and convert the d dictionary values to global variables.
    :param config_file: Path to the config.yaml file.
    """
    # Read the YAML file
    assert isinstance(d, dict)

    # Convert values to appropriate types (int or float)
    for key, value in d.items():
        if isinstance(value, str) and 'e' in value.lower():  # Check for scientific notation
            d[key] = float(value)  # Convert to float
        elif isinstance(value, str) and value.isdigit():  # Check for integer strings
            d[key] = int(value)  # Convert to int
        elif isinstance(value, str) and '.' in value:  # Check for float strings
            d[key] = float(value)  # Convert to float

    # Convert dictionary values to global variables
    globals().update(d)

