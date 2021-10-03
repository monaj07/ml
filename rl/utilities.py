import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch


def make_deterministic(internal_seed, env=None):
    # ----------------------------------------
    # Make the algorithm outputs reproducible
    torch.manual_seed(internal_seed)
    np.random.seed(internal_seed)
    random.seed(internal_seed)
    torch.cuda.manual_seed_all(internal_seed)
    torch.cuda.manual_seed(internal_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if env is not None:
        env.seed(internal_seed)
        env.action_space.seed(internal_seed)
    # ----------------------------------------


def plot_durations(episode_durations, rolling_reward, log_tag):
    # plt.figure(2)
    # plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label='Single episode reward')
    # Take 100 episode averages and plot them too
    plt.plot(rolling_reward, label='Average reward in the last 100 episodes')
    plt.legend()
    plt.savefig(f'./logs_and_figs/{log_tag}')
    plt.pause(0.001)  # pause a bit so that plots are updated
