# -- Public Imports
from gym import Env, spaces
import yaml
import random
import itertools
import numpy as np

# -- Private Imports
from utils import *

# -- Global Variables
bits_per_byte = 8
packet_size = 32   # bytes

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -- Functions

# Define Resource Block class
class RB:
    """
    Resource Block contains info
    """
    def __init__(self, rth, kth=-1, mth=-1, is_allocated=False):
        self.rth = rth   # the idx of the RB
        self.kth = kth   # the idx of the BS that the RB is allocated
        self.mth = mth   # the idx of the UE that the RB is allocated
        self.is_allocated = is_allocated

    def __str__(self):
        """
        Returns a string representation of the RB object.
        """
        return (
            f"RB(rth={self.rth}, kth={self.kth}, mth={self.mth}, "
            f"is_allocated={self.is_allocated})"
        )

    def update_params(self, kth, mth, is_allocated):
        self.kth = kth
        self.mth = mth
        self.is_allocated = is_allocated

    def reset(self):
        self.kth = -1
        self.mth = -1
        self.is_allocated = False


class UE:
    """
    User Equipment Class
    """
    def __init__(self, x, y, speed, traffic_type, num_rbs_allocated=0):

        assert traffic_type in ('tcp_full', 'udp_bursty', 'tcp_bursty')

        # position info
        self.x = x
        self.y = y
        self.speed = speed          # m/s
        self.direction = np.random.uniform(0, 2*np.pi)
        self.traffic_type = traffic_type
        self.data_rate = self.set_data_rate()
        self.is_active = False      # For bursty traffic: whether UE is in "on" phase
        self.next_switch_time = 0   # Time until the next "on" or "off" time

        # data rate / service rate
        self.service_rate = 0       # (bps) Current service rate depending on the num_rbs_allocated
        self.service_rate_avg = 0   # (bps) Averaged service rate for the last 10 sec

        self.delay = 0                             # (sec) Total delay for the current packet (sec)
        self.num_rbs_allocated = num_rbs_allocated
        self.rbs_indices = []

        # Store the last 10 traffic (last 10 seconds)
        self.service_rate_history = [0]

    def set_data_rate(self):
        if self.traffic_type in ('tcp_full', 'udp_bursty'):
            return 20e6
        else:   # tcp_bursty
            return random.choice([750e3, 150e3])

    def move(self, dt):
        dx = self.speed * np.cos(self.direction) * dt
        dy = self.speed * np.sin(self.direction) * dt
        self.x += dx
        self.y += dy

        self.direction += np.random.uniform(-0.1*np.pi, 0.1*np.pi)

    def update_traffic(self, current_time):
        if self.traffic_type in ("udp_bursty", "tcp_bursty"):
            if current_time >= self.next_switch_time:
                self.is_active = not self.is_active # Switch between on and off
                self.next_switch_time = current_time + np.random.exponential(5)
        else:
            # TCP full-buffer traffic is always active
            self.is_active = True

    def update_service_rate(self, Ckm):
        self.service_rate = Ckm
        self.service_rate_history.append(self.service_rate)
        if len(self.service_rate_history) > 10:
            self.service_rate_history.pop(0)

        self.service_rate_avg = sum(self.service_rate_history) / len(self.service_rate_history)

    def reset(self):
        """Reset the UE state"""
        self.delay = 0
        self.num_rbs_allocated = 0
        self.rbs_indices = []


# Define gNB class
class gNB:
    """
    gNB contains 9 UEs
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.UEs = []

    def add_ue(self, ue):
        self.UEs.append(ue)


# Define O-RAN class
class ORAN:
    """
    ORAN contains 7 gnbs
    """
    def __init__(self):
        self.num_gnbs = config.get('oran_env').get('num_gnb')
        self.radius = config.get('oran_env').get('inter_distance_gnb')

        # Define cellular network
        self.gNBs = self.get_init_gnbs()

    def get_init_gnbs(self):
        gNBs = [gNB(2000, 2000)]
        for idx_gnb in range(self.num_gnbs):
            angle = 2 * np.pi * idx_gnb / 6
            x = 2000 + self.radius * np.cos(angle)
            y = 2000 + self.radius * np.sin(angle)
            gNBs.append(gNB(x, y))

        return gNBs

    def set_init_ues_per_gnb(self):
        for gnb in self.gNBs:
            for idx_ue in range(9):
                # Random init position
                x = gnb.x + np.random.uniform(-100, 100)
                y = gnb.y + np.random.uniform(-100, 100)

                # Random speed
                speed = np.random.uniform(2, 4)

                # Assign traffic type based on distributions
                if idx_ue < 2:
                    traffic_type = 'tcp_full'
                elif idx_ue < 4:
                    traffic_type = 'udp_bursty'
                else:
                    traffic_type = 'tcp_bursty'

                ue = UE(x, y, speed, traffic_type)
                gnb.add_ue(ue)
