"""
Implementing a DQN agent.
"""
from collections import Counter
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import Base_Agent
from exploration_strategies.epsilon_greedy_exploration import EpsilonGreedyExploration
from utilities.data_structures.replay_buffer import Replay_Buffer
from utilities.networks import CNN, DenseTwoLayer


class DQN(Base_Agent):
    """A deep Q learning agent"""
    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(
            self.hyperparameters["buffer_size"],
            self.hyperparameters["batch_size"],
            config.seed,
            self.device
        )
        self.q_network = self.create_neural_net(
            input_dim=self.state_size,
            output_dim=self.action_size
        )
        self.q_network_optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.hyperparameters["learning_rate"],
            eps=1e-4
        )
        self.exploration_strategy = EpsilonGreedyExploration(config)

    def create_neural_net(self, input_dim, output_dim):
        """Creates a neural network for the agents to use"""
        if isinstance(input_dim, tuple):
            QNetwork = CNN
        else:
            QNetwork = DenseTwoLayer
        model = QNetwork(input_dim, output_dim).to(self.device)
        return model

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(
            self.hyperparameters["learning_rate"],
            self.q_network_optimizer
        )

    def run_one_episode(self):
        """Runs an episode within a game including learning steps if required"""
        while not self.done:
            self.action = self.pick_action()
            self.take_action(self.action)
            update_freq = (
                    self.global_step_number % self.hyperparameters["update_every_n_steps"]
            ) == 0
            sufficient_replay_memory = (len(self.memory) > self.hyperparameters["batch_size"])
            if update_freq and sufficient_replay_memory:
                # Go through a learning iteration
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn()
            if self.get_visual_state is not None:
                self.next_state = self.get_visual_state(self)
            self.save_experience()
            self.state = self.next_state  # this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None:
            state = self.state
        if isinstance(state, np.int64) or isinstance(state, int):
            state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2:
            state = state.unsqueeze(0)
        self.q_network.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train() #puts network back in training mode
        action, self.epsilon = self.exploration_strategy.perturb_action_for_exploration_purposes(
            {"action_values": action_values,
             "episode_number": self.episode_number,
             "global_step_number": self.global_step_number}
        )
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None:
            states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions ]

        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(
            self.q_network_optimizer,
            self.q_network,
            loss,
            self.hyperparameters["gradient_clipping_norm"]
        )

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = nn.functional.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """Computes the q_targets we will compare to predicted q values to create the loss to train the Q network"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """Computes the q_values for next state we will use to create the loss to train the Q network"""
        Q_targets_next = self.q_network(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """Computes the q_values for current state we will use to create the loss to train the Q network"""
        Q_targets_current = rewards.unsqueeze(1) + (
                self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones.unsqueeze(1))
        )
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """Computes the expected q_values we will use to create the loss to train the Q network"""
        Q_expected = self.q_network(states).gather(1, actions.unsqueeze(1).long()) #must convert actions to long so can be used as index
        return Q_expected

    def locally_save_policy(self):
        """Saves the policy"""
        torch.save(self.q_network.state_dict(), "Models/{}_network.pt".format(self.agent_name))

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
