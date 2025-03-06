import argparse
import config

# Libs used
import numpy as np
import random

# Modules used
import graphGenerator
import rateMapping
import resultRecorder

# Configs
SLICE_QOS = config.sys_config.SLICE_QOS
P_NOISE = config.sys_config.P_NOISE
N_PRB = config.sys_config.N_PRB
OUTAGE_RATE = config.OUTAGE_RATE
SLOT_NUM = 100
SIGMA = 3

def randomSchedulePRBs(base_station, node, all_slicing_policy, SLICE_QOS, PRB):
    N_BS = len(base_station)
    N_UE = len(node)
    SLICE_NUM = len(all_slicing_policy[0])

    # Specifiy each user's assosiated slice and BS
    ue_slice = []
    for bs_id in range(N_BS):
        us_slice_in_bs = [[] for _ in range(SLICE_NUM)]
        for i in range(len(base_station[bs_id].node_ids)):
            ue_id = base_station[bs_id].node_ids[i]
            us_slice_in_bs[node[ue_id].slice_id].append(ue_id)
        ue_slice.append(us_slice_in_bs)

    # Initialzation
    resource_usage = np.zeros((SLOT_NUM,))
    qos_outage = np.zeros((SLOT_NUM, N_UE))
    network_throughput = np.zeros((SLOT_NUM,))

    # Start simulation for each timestep
    for t in range(SLOT_NUM):

        # Use an algorithm to allocate PRBs to each user
        prb_used_slice = np.zeros((N_BS, SLICE_NUM))
        remain_prb_num = PRB
        for bs_id in range(N_BS):
            remain_prb_num = PRB
            for slice_no in base_station[bs_id].deployed_slice:
                prb_mu = all_slicing_policy[bs_id][slice_no]*PRB
                prb_num = int(random.gauss(prb_mu, SIGMA))
                prb_num = max(prb_num, 0)
                prb_num = min(prb_num, remain_prb_num)
                prb_used_slice[bs_id, slice_no] = prb_num
                remain_prb_num = max(remain_prb_num - prb_num, 0)

        prb_used_ue = np.zeros((N_UE,))
        for bs_id in range(N_BS):
            for slice_no in base_station[bs_id].deployed_slice:
                ue_ids = ue_slice[bs_id][slice_no]
                if len(ue_ids) != 0:
                    prb_per_ue = int(prb_used_slice[bs_id][slice_no]/len(ue_ids))
                    remain_prb = prb_used_slice[bs_id][slice_no] - prb_per_ue*len(ue_ids)
                    for ue_id in ue_ids:
                        prb_used_ue[ue_id] = prb_per_ue
                    prb_used_ue[random.sample(ue_ids, 1)] += remain_prb

        x = np.zeros((N_UE, PRB))
        prb_allocate_2_ue = [[] for _ in range(N_UE)]
        for bs_id in range(N_BS):
            allocate = np.array(list(range(PRB)))
            allocate = np.random.permutation(allocate)
            counter = 0
            for ue_id in base_station[bs_id].node_ids:
                for _ in range(int(prb_used_ue[ue_id])):
                    prb_allocate_2_ue[ue_id].append(int(allocate[counter]))
                    x[ue_id, int(allocate[counter])] = 1
                    counter += 1
        
        # Get all the BS's total utilization of PRBs (number of PRBs)
        resource_usage[t] = np.sum(x)

        # Compute SINR on each PRB --> CQI (MCS)--> Rate
        rate = np.zeros((N_UE,))
        for ue_id in range(N_UE):
            for prb in prb_allocate_2_ue[ue_id]:
                p_i = 0
                for i in range(len(node[ue_id].neigh_ids)):
                    neigh_ue_id = node[ue_id].neigh_ids[i]
                    if prb in prb_allocate_2_ue[neigh_ue_id]:
                        p_i += 10**(np.divide(node[ue_id].p_i[i], 10))

                prb_SINR = 10 * np.log10((10**(np.divide(node[ue_id].p_s, 10)))/(P_NOISE + p_i))
                prb_CQI = rateMapping.snr_to_CQI(prb_SINR)
                prb_rate = rateMapping.CQI_to_rate[prb_CQI]
                rate[ue_id] += prb_rate

            qos_outage[t, ue_id] = rate[ue_id] - SLICE_QOS[node[ue_id].slice_id]
            network_throughput[t] += rate[ue_id]


    # 针对SLOT_NUM次实验，将统计量们取时间上的平均
    resource_usage = np.mean(resource_usage, axis = 0)
    qos_outage = np.mean(qos_outage, axis = 0)
    network_throughput = np.mean(network_throughput, axis = 0)
    
    return resource_usage, qos_outage, network_throughput
