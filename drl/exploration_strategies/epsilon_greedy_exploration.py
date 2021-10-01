from .base_exploration_strategy import BaseExplorationStrategy
import numpy as np
import random
import torch


class EpsilonGreedyExploration(BaseExplorationStrategy):
    """Implements an epsilon greedy exploration strategy"""
    def __init__(self, config):
        super().__init__(config)

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_values = action_info["action_values"]
        epsilon = self.get_updated_epsilon_exploration(action_info)

        if random.random() > epsilon:
            return torch.argmax(action_values).item()
        return np.random.randint(0, action_values.shape[1])

    def get_updated_epsilon_exploration(self, action_info, epsilon=1.0):
        """Gets the probability that we just pick a random action.
        This probability decays the more episodes we have seen"""
        episode_number = action_info["episode_number"]
        epsilon_decay_denominator = self.config.hyperparameters["epsilon_decay_rate_denominator"]

        epsilon = epsilon / (1.0 + (episode_number / epsilon_decay_denominator))
        return epsilon

    def reset(self):
        """Resets the noise process"""
        pass