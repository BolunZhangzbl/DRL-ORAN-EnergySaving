# -- Public Imports
import random

# -- Private Imports
from utils import *
from constants import *

# -- Global Variables


# -- Functions

# Define Resource Block class
class RB:
    """
    Resource Block contains info
    """
    def __init__(self, rth):
        self.rth = rth   # the idx of the RB
        self.allocate_to = None      # UE to which the RB is allocated to

    def __str__(self):
        """
        Returns a string representation of the RB object.
        """
        return (
            f"RB(rth={self.rth}, allocate_to={self.allocate_to})"
        )

    def update_params(self, allocate_to):
        self.allocate_to = allocate_to

    def reset(self):
        self.allocate_to = None


class UE:
    """
    User Equipment (UE) class.
    """

    def __init__(self, x: float, y: float, speed: float, traffic_type: str, serving_gnb):
        """
        Initialize a UE object.
        :param x: Initial x-coordinate of the UE.
        :param y: Initial y-coordinate of the UE.
        :param speed: Speed of the UE in meters per second.
        :param traffic_type: Type of traffic ('tcp_full', 'udp_bursty', or 'tcp_bursty').
        :param serving_gnb: The gNB serving this UE.
        """
        assert traffic_type in ('tcp_full', 'udp_bursty', 'tcp_bursty')

        # Position info
        self.x = x
        self.y = y
        self.speed = speed          # m/s
        self.direction = np.random.uniform(0, 2*np.pi)

        # Traffic info
        self.traffic_type = traffic_type
        self.data_rate = self.set_data_rate()   # Demanding traffic
        self.is_active = False      # For bursty traffic: whether UE is in "on" phase
        self.next_switch_time = 0   # Time until the next "on" or "off" time

        # Delay info
        self.buffer_size = 0
        self.delay = 0
        self.delay_max = 10   # sec

        # PRB allocation
        self.num_rbs_allocated = 0
        self.service_rate = 0  # (bps) Current service rate depending on the num_rbs_allocated
        self.serving_gnb = serving_gnb

        # RLF
        self.sinr = -10             # dB
        self.rlf_threshold = -5     # dB
        self.in_rlf = False         # Binary indicator whether the current UE is in Radio Failure Link

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

    def update_service_rate(self):
        self.service_rate = self.num_rbs_allocated * RB_EFFICIENCY

    def update_delay(self):
        """
        Update the delay based on the buffer size and service rate.
        :param time_interval: Time interval in seconds.
        """

        # Update buffer size based on data rate
        if self.is_active:
            self.buffer_size += self.data_rate * TIME_STEP

        # Process buffer based on service rate
        if self.service_rate > 0:
            self.buffer_size = max(self.buffer_size - self.service_rate * TIME_STEP, 0)

        # Calculate delay
        if self.service_rate > 0:
            self.delay = self.buffer_size / self.service_rate
        else:
            self.delay = self.delay_max

    def get_sinr(self, gNBs):
        # Constants
        power_noise = BANDWIDTH_PER_RB * NOISE_POWER_DENSITY   # Noise power in watts

        # Calculate received power from the serving gNB
        serving_gNB = self.serving_gnb
        distance = np.sqrt((self.x - serving_gNB.x)**2 + (self.y - serving_gNB.y)**2)
        path_loss = self.get_path_loss(distance)
        power_rx_dbm = serving_gNB.power_tx - path_loss

        # Convert received power to linear scale watts
        power_rx = 10 ** ((power_rx_dbm - 30) / 10)

        # Calculate interference from neighbouring gNBs
        power_interference = 0
        for gNB in gNBs:
            if gNB != serving_gNB:
                distance = np.sqrt((self.x - gNB.x)**2 + (self.y - gNB.y)**2)
                path_loss = self.get_path_loss(distance)
                power_interference_dbm = gNB.power_tx - path_loss
                power_interference += 10 ** ((power_interference_dbm - 30) / 10)

        # Calculate SINR
        sinr = power_rx / (power_interference + power_noise)

        # Update RLF info
        self.in_rlf = True if sinr < self.rlf_threshold else False
        self.sinr = sinr

        return sinr

    def get_path_loss(self, distance):

        path_loss = 28.0 + 40.0 * np.log10(distance) + 20 * np.log10(CENTER_FREQ / 1e9)

        # Add log-normal shadowing
        shadowing = np.random.uniform(0, SHADOWING_STD)

        path_loss += shadowing

        return path_loss

    def reset(self):
        """Reset the UE state"""
        self.buffer_size = 0
        self.delay = 0
        self.num_rbs_allocated = 0
        self.serving_gnb = None
        self.in_rlf = False


# Define gNB class
class gNB:
    """
    gNB contains 9 UEs
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.UEs = []

        # Power consumption model
        self.P0 = 50  # Baseline power consumption (W)
        self.PRF = 20  # RF power consumption (W)
        self.PBB = 30  # Baseband power consumption (W)
        self.PPAM_0 = 10  # Base power consumption of PAM (W)
        self.eta_PAM = 0.8  # Efficiency of the power amplifier
        self.n_ant = 32  # Number of antenna chains

        # Default transmission power (watts)
        self.power_tx = 40

        # Activation info
        self.is_active = False
        self.active_time_interval = 0   # Total active_time_interval in sec

        # KPMs
        self.throughput = 0
        self.num_ues_rlf = 9
        self.num_rbs_allocated = 0
        self.activate_cost = 0

    def add_ue(self, ue):
        self.UEs.append(ue)

    def update_active_status(self, flag_active):
        """
        Update the gNB's activation status and active time interval.
        :param flag_active: Boolean indicating whether the gNB is active.
        """
        if self.is_active != flag_active:
            self.active_time_interval = 0
        else:
            if self.is_active:
                self.active_time_interval += TIME_STEP

        self.is_active = True if flag_active else False

        # if the gNB is inactive, reset relevant metrics
        if not self.is_active:
            self.num_rbs_allocated = 0
            for ue in self.UEs:
                ue.num_rbs_allocated = 0
                ue.service_rate = 0
                ue.buffer_size = 0
                ue.delay = 0
                ue.in_rlf = True   # Mark UE as in RLF due to gNB inactivity

    def update_num_rbs_allocated(self):
        self.num_rbs_allocated = sum(ue.num_rbs_allocated for ue in self.UEs)

    def set_power_tx(self):

        self.power_tx = self.power_tx * (self.num_rbs_allocated / NUM_RBS)

    def get_throughput(self):
        """
        Calculate the total throughput of the cell (sum of service rates of all UEs)
        :return:
        """
        throughput = sum(ue.service_rate for ue in self.UEs)

        return throughput

    def get_power_consumption(self):
        """
        Calculate the total power consumption of RU (watts)
        :return:
        """
        # Calculate PPAM_c(r) based on the number of RBs
        PPAM_c = self.get_PPAM_c(self.num_rbs_allocated)

        # Calculate PPAM
        PPAM = self.PPAM_0 * PPAM_c * self.eta_PAM * self.n_ant

        # Calculate total power consumption
        P_total = self.P0 + self.PRF + self.PBB + PPAM

        return P_total

    def get_PPAM_c(self, num_rbs_allocated):
        PPAM_c = num_rbs_allocated / NUM_RBS

        return PPAM_c

    def get_ratio_throughput_energy_consumption(self):
        throughput = self.get_throughput()
        power_consumption = self.get_power_consumption()

        ratio = throughput / power_consumption

        return ratio

    def get_num_ues_rlf(self):
        num_ues_rlf = 0
        for ue in self.UEs:
            num_ues_rlf += int(ue.in_rlf)

        return num_ues_rlf

    def get_percentage_rlf(self, num_ues_rlf=None):
        num_ues_total = len(self.UEs)
        num_ues_rlf = self.get_num_ues_rlf() if num_ues_rlf==None else num_ues_rlf

        percentage_rlf = num_ues_rlf / num_ues_total

        return percentage_rlf

    def get_percentage_scheduled_rbs(self):
        num_scheduled_rbs = self.num_rbs_allocated

        percentage_scheduled_rbs = num_scheduled_rbs / NUM_RBS

        return percentage_scheduled_rbs

    def get_activate_cost(self):
        active_time_interval_ms = self.active_time_interval * 1e3
        activate_cost = 0.9 ** (0.01 * active_time_interval_ms)

        return activate_cost

    def get_reward(self):
        """
        Calculate the reward value given the weight distributions
        :return:
        """
        w1, w2, w3, w4 = 0.4, 0.4, 0.1, 0.1

        if self.is_active == True:
            throughput = self.get_throughput()
            power_consumption = self.get_power_consumption()
            num_ues_rlf = self.get_num_ues_rlf()
            activate_cost = self.get_activate_cost()

            reward = w1 * throughput - w2 * power_consumption - w3 * num_ues_rlf - w4 * activate_cost
            return reward

        else:
            return 0

    def get_state_gnb(self):
        """
        The state info contains KPMs:
        1. Ratio between throughput and energy consumption at cell i
        2. Number of UEs in RLF at cell i
        3. Percentage of UEs in RLF at cell i
        4. Number of scheduled PRBs at cell i
        5. Percentage of scheduled PRBs at cell i
        6. Cost to activate at cell i
        7. Tx Power at cell i
        :return:
        """
        # Calculate KPMs
        ratio_throughput_energy = self.get_ratio_throughput_energy_consumption()
        num_ues_rlf = self.get_num_ues_rlf()
        percentage_rlf = self.get_percentage_rlf(num_ues_rlf)
        num_scheduled_rbs = self.num_rbs_allocated
        percentage_scheduled_rbs = self.get_percentage_scheduled_rbs()
        activate_cost = self.get_activate_cost()
        power_tx = self.power_tx

        state_gnb = [
            ratio_throughput_energy, num_ues_rlf, percentage_rlf, num_scheduled_rbs, percentage_scheduled_rbs,
            activate_cost, power_tx
         ]

        return state_gnb


# Define O-RAN class
class ORAN:
    """
    ORAN contains 7 gnbs
    """
    def __init__(self):
        self.num_gnbs = NUM_GNB
        self.radius = INTER_DISTANCE_GNB

        # Define cellular network
        self.gNBs = self.get_init_gnbs()
        self.set_init_ues_per_gnb()

        # Define PRBs
        self.RBs = [RB(rth=idx) for idx in range(NUM_RBS)]

        # Active status for all gNBs
        self.gnbs_active_status = []

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
            for idx_ue in range(NUM_UES_PER_GNB):
                # Random init position within 1700m radius of the central gNB
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, self.radius)
                x = 2000 + radius * np.cos(angle)
                y = 2000 + radius * np.sin(angle)

                # Random speed
                speed = np.random.uniform(2, 4)

                # Assign traffic type based on distributions
                if idx_ue < 2:   # 25%
                    traffic_type = 'tcp_full'
                elif idx_ue < 4: # 25%
                    traffic_type = 'udp_bursty'
                else:            # 50%
                    traffic_type = 'tcp_bursty'

                ue = UE(x, y, speed, traffic_type, gnb)
                gnb.add_ue(ue)

    def update_system(self, current_time, gnbs_active_status):
        """
        Update the entire ORAN system at each time step
        :param current_time:
        :param active_status:
        :return:
        """
        assert len(gnbs_active_status) == NUM_GNB

        # Update active status for all gNBs
        self.gnbs_active_status = gnbs_active_status

        # Update gNBs and their UEs
        for idx_gnb, gnb in enumerate(self.gNBs):
            # Update active status for each gNB
            gnb.update_active_status(gnbs_active_status[idx_gnb])

            for ue in gnb.UEs:
                # Update UE position
                ue.move(TIME_STEP)

                # Update UE traffic status
                ue.update_traffic(current_time)

                # Update UE's SINR
                ue.get_sinr(self.gNBs)

                # Update service rate based on allocated RBs
                ue.update_service_rate()

                # Update delay based on buffer_size and service_rate
                ue.update_delay()

        # Recycle unused RBs
        self.recycle_rbs()

        # Allocate RBs according to proportional policy
        self.allocate_rbs()

        # Update the number of RBs allocated for each gNB
        for gnb in self.gNBs:
            gnb.update_num_rbs_allocated()

    def get_proportional_weights(self):
        """
        Calculate proportional weights for all UEs
        :return:
        """
        priority_traffic_type = dict(udp_bursty=2, tcp_bursty=1, tcp_full=1.5)

        proportional_weights = {}
        total_weighted_buffer = 0

        for gnb in self.gNBs:
            for ue in gnb.UEs:
                weight = 1 + ue.sinr
                priority = priority_traffic_type.get(ue.traffic_type)

                weighted_buffer = ue.buffer_size * weight * priority
                proportional_weights[ue] = weighted_buffer
                total_weighted_buffer += weighted_buffer

        # Normalize weights
        if total_weighted_buffer > 0:
            for ue in proportional_weights:
                proportional_weights[ue] /= total_weighted_buffer

        return proportional_weights

    def allocate_rbs(self):
        """
        Allocate RBs to UEs based on their proportional weights.
        :return:
        """
        rbs_available = [rb.rth for rb in self.RBs if rb.allocate_to==None]
        num_rbs_available = len(rbs_available)
        proportional_weights = self.get_proportional_weights()

        # Allocate RBs to each UE based on its proportional weight
        for ue, weight in proportional_weights.items():
            num_rbs_allocated = int(weight * num_rbs_available)
            for idx in range(num_rbs_allocated):
                ue.num_rbs_allocated += 1
                self.RBs[rbs_available[0]].allocate_to = ue
                rbs_available.pop(0)

    def recycle_rbs(self):
        """
        Recycle unused RBs
        :return:
        """
        for rb in self.RBs:
            if rb.allocate_to is not None and rb.allocate_to.buffer_size == 0:
                rb.allocate_to.num_rbs_allocated -= 1
                rb.allocate_to = None

    def get_sum_reward(self):
        """
        Calculate sum reward for all gNBs at current time step
        :return:
        """
        sum_reward = 0
        for gnb in self.gNBs:
            sum_reward += gnb.get_reward()

        return sum_reward

    def get_state_oran(self):
        """
        :return: state info for all the gNBs
        """
        state_oran = []

        # Add gnb state
        for gnb in self.gNBs:
            state_oran += gnb.get_state_gnb()

        # Add global state
        state_oran += [sum(1 for status in self.gnbs_active_status if status)]

        return state_oran