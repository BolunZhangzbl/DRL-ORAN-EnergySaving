# -- Public Imports
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -- Private Imports
from utils import *
from constants import *

# -- Global Variables

dir_root = str(Path(__file__).parent.parent)

dict_ylabel = dict(
    step_losses="Loss",
    ep_losses='Episodic Loss',
    step_rewards='Reward',
    avg_rewards='Avg. Reward',
    ep_rewards='Episodic. Reward'
)

dict_markers = dict(
    step_losses='^-',
    ep_losses='o--',
    step_rewards='s-',
    avg_rewards='D--',
    ep_rewards='*-'
)

dict_colors = dict(
    step_losses='blue',
    ep_losses='green',
    step_rewards='red',
    avg_rewards='orange',
    ep_rewards='black'
)


# -- Functions

def plot_convergence(metric, agent_type, save=False):

    assert metric in dict_ylabel.keys()

    file_path = os.path.join(dir_root, f"lists/{agent_type}/training_metrics.npz")
    dict_data = np.load(file_path)
    data = dict_data.get(metric)
    xaxis = np.arange(len(data))
    print(data)

    plt.figure(figsize=(15, 10))
    plt.semilogy(xaxis, data, dict_markers.get(metric), color=dict_colors.get(metric),
                 mfc='none', alpha=0.8, lw=2, markersize=3, label=metric.upper())

    plt.yscale('linear')
    plt.xlabel("Episode" if metric.startswith('ep') else "Step", fontsize=30)
    plt.ylabel(dict_ylabel.get(metric), fontsize=30)
    # plt.xlim([1, 1000])
    # plt.ylim([-400, 200])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=27)
    plt.grid(True, which='both', linestyle='--')

    if save:
        filename_save = f"{metric}_convergence_{agent_type}.png"
        filepath_save = os.path.join("./plots", "figures", filename_save)
        plt.savefig(filepath_save, format='png', dpi=300)

    plt.show()


plot_convergence("step_losses", "dqn", save=False)