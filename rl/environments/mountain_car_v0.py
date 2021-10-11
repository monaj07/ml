"""
Mountain_car environment class.
"""
import gym
import numpy as np
from os.path import dirname, abspath
from PIL import Image
import sys
import torch
import torchvision.transforms as T

sys.path.append(dirname(dirname(abspath(__file__))))
from utilities import make_deterministic


class MountainCarV0:
    """
    This class instantiate the environment.
    """
    def __init__(self, seed=1364):

        # Define the environment
        self.env = gym.make('MountainCarContinuous-v0').unwrapped
        # ----------------------------------------
        # Make the algorithm outputs reproducible
        make_deterministic(seed, self.env)
        # ----------------------------------------
        self.env.reset()

        # Get number of actions from gym action space
        self.action_dimension = self.env.action_space.shape[0]
        # Get the space size
        self.input_dim = self.env.state.size
        self.score_required_to_win = 90
        self.average_score_required_to_win = self.env.spec.reward_threshold

    def get_state(self, episode_start=False):
        state = np.array(self.env.state.squeeze())
        state = torch.from_numpy(state).unsqueeze(0).float()
        return state


if __name__ == "__main__":
    seed = 1364
    mountain_car = MountainCarV0(seed=seed)
