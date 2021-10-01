"""
This is the primary file for running the implemented agents
in the Cart-Pole environment.
"""
import gym
import numpy as np
from os.path import dirname, abspath
from PIL import Image
import sys
import torch
import torchvision.transforms as T
from types import SimpleNamespace
sys.path.append(dirname(dirname(abspath(__file__))))
# -------------------------------------
from agents.trainer import Trainer
from agents.dqn import DQN
from utilities.utility_functions import get_screen


def get_visual_state(self, episode_start=False):
    if episode_start:
        self.prev_screen = get_screen(self.environment)
    self.curr_screen = get_screen(self.environment)
    preprocessed_state = (self.curr_screen - self.prev_screen).copy()
    self.prev_screen = self.curr_screen
    return preprocessed_state


config = SimpleNamespace()
# config.seed = 1
config.environment = gym.make("CartPole-v0")
config.num_episodes_to_run = 501
config.file_to_save_data_results = "results/data_and_graphs/Cart_Pole_Results_Data.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/Cart_Pole_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False

# config.get_visual_state = get_visual_state
config.get_visual_state = None

config.hyperparameters = {
    "DQN": {
        "learning_rate": 0.01,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 1.0,
        "use_epsilon_exponential_decay": False,
        "epsilon_exp_decay_coef": 0.05,
        "epsilon_decay_rate_denominator": 1,
        "discount_rate": 0.99,
        "update_every_n_steps": 1,
        "gradient_clipping_norm": 0.7,
        "learning_iterations": 1,
        "clip_rewards": False
    }
}

if __name__ == "__main__":
    AGENTS = [
        DQN
    ]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_all_agents()
