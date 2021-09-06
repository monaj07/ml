import random
from skimage import transform
from skimage.color import rgb2gray
import torch
from torch import nn
import torch.nn.functional as F
import time
from vizdoom import *
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')


def create_doom_env():
    game = DoomGame()
    game.load_config(
        "../../../AppData/Local/Continuum/miniconda3/envs/py372/Lib/site-packages/vizdoom/scenarios/basic.cfg")
    game.set_doom_scenario_path(
        "../../../AppData/Local/Continuum/miniconda3/envs/py372/Lib/site-packages/vizdoom/scenarios/basic.wad")
    game.init()

    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    possible_actions = [shoot, left, right]

    return game, possible_actions


"""
# Test if the vizdoom environment and `create_doom_env()` function work:
def test_environment():
    game, actions = create_doom_env()
    episodes = 3
    for ep in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            game_vars = state.game_variables
            action = random.choice(actions)
            reward = game.make_action(action)
            print(reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
    game.close()


test_environment()
"""


class BuildQNet(nn.Module):
    def __init__(self, num_actions=3, stack_num_channels=4):
        super(BuildQNet, self).__init__()
        self.conv_1 = nn.Conv2d(stack_num_channels, 32, 8, stride=2, padding=0)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 128, 4, stride=2, padding=0)
        self.bn_3 = nn.BatchNorm2d(128)
        self.fc_1 = nn.Linear(8192, 512)
        self.fc_2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # conv + BN + ELU
        x1 = self.conv_1(x)
        x1 = F.elu(self.bn_1(x1))
        # conv + BN + ELU
        x2 = self.conv_2(x1)
        x2 = F.elu(self.bn_2(x2))
        # conv + BN + ELU
        x3 = self.conv_3(x2)
        x3 = F.elu(self.bn_3(x3))
        # flatten layer
        x3 = torch.flatten(x3, start_dim=1)
        # dense + ELU
        x4 = F.elu(self.fc_1(x3))
        # output dense layer
        x5 = self.fc_2(x4)
        return x5


class DQNetwork:
    def __init__(self):
        # whether the second part of the reward (max{Q(s', a'; theta)}) is computed earlier
        # during observations (theta_constant=True),
        # or using current parameter that is used for Q(s, a).
        self.theta_constant = True

        # Exploration parameters for epsilon greedy strategy
        self.epsilon_start = 1.0  # exploration probability at start
        self.epsilon_stop = 0.01  # minimum exploration probability
        self.decay_rate = 0.0001  # exponential decay rate for exploration prob

        # TRAINING HYPERPARAMETERS
        self.total_episodes = 500  # Total episodes for training
        self.max_steps = 100  # Max possible steps in an episode
        self.batch_size = 64
        self.training = True
        self.current_episode = 0
        self.is_new_episode = True

        # Q learning hyperparameters
        self.gamma = 0.95  # Discounting rate
        self.learning_rate = 0.0002  # Alpha (aka learning rate)

        # MEMORY HYPERPARAMETERS
        self.memory_size = 1000000  # Number of experiences the Memory can keep
        self.experience_box = deque([], maxlen=self.memory_size)

        # Input properties
        self.stack_size = 4
        self.input_size = (84, 84)
        self.stacked_frames = deque([], maxlen=self.stack_size)

        # Initialising the game environment
        self.game, self.possible_actions = create_doom_env()
        self.action_size = self.game.get_available_buttons_size()

        # Building the Q-Network
        self.model = BuildQNet(self.action_size, self.stack_size)

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.99,
            weight_decay=0.0005
        )

        for episode in range(self.total_episodes):
            # Start a new episode
            self.game.new_episode()

            # Set the value of epsilon for explore/exploit trade-off
            epsilon = self.epsilon_stop + (
                    (self.epsilon_start - self.epsilon_stop) *
                    (1 - self.decay_rate) ** episode
            )

            # self.new_episode is used to check if we are at the start of the spisode,
            # in which case the 4-channel stacked_frames is instantiated, using four
            # copies of the first frame of the episode (in the _stack_frames method):
            self.is_new_episode = True

            for _ in range(self.max_steps):
                # Append the transition (s, a, r, s') that is observed
                # in the episode to the experience_box.
                state = self.game.get_state()
                img = state.screen_buffer

                # Produce a transition
                state_1 = self._stack_frames(img)
                state_1_tensor = torch.tensor(state_1, dtype=torch.float32).unsqueeze(0)
                action_1 = self._get_epsilon_greedy_action(state_1_tensor, epsilon)
                reward_1 = self.game.make_action(self.possible_actions[action_1])
                state_2 = self._stack_frames(self.game.get_state().screen_buffer)
                next_state_is_terminal = self.game.is_episode_finished()
                if self.theta_constant:
                    # The full reward R = r + gamma * max_a2{Q(s2, a2; theta)} is computed here
                    # during observation collection
                    if not self.game.is_episode_finished():
                        # If it is in a non-terminal step
                        state_2_tensor = torch.tensor(state_2, dtype=torch.float32).unsqueeze(0)
                        reward_1 += self.gamma * self.model(state_2_tensor).max().item()

                # Pack the transition into a tuple
                transition = (state_1, action_1, reward_1, state_2, next_state_is_terminal)

                # Append the transition to the experience replay box
                self.experience_box.append(transition)

                # Sample a random minibatch of transitions from experience_box,
                # if it is large enough.
                if len(self.experience_box) >= self.batch_size:
                    # Reset the gradients
                    optimizer.zero_grad()

                    # Create a minibatch
                    minibatch = np.random.choice(self.experience_box, self.batch_size, replace=False)

                    # Calculate the predictions for Q(s1, a1; theta)
                    s1_tensor = np.stack([transition[0] for transition in minibatch], axis=0)
                    s1_qvalue = self.model(s1_tensor)
                    target_reward_tensor = np.stack([transition[2] for transition in minibatch], axis=0)

                    if not self.theta_constant:
                        # Calculate the target rewards: R = r + gamma * max_a2{Q(s2, a2; theta)}, OR R = r
                        # Depending on whether we are in terminal state or not
                        s2_is_terminal_tensor = np.stack([transition[4] for transition in minibatch], axis=0)
                        s2_tensor = np.stack([transition[3] for transition in minibatch], axis=0)
                        s2_max_qvalue = self.model(s2_tensor).max(dim=-1)
                        target_reward_tensor = target_reward_tensor + (self.gamma *
                                                                       s2_is_terminal_tensor *
                                                                       s2_max_qvalue)

                    loss = F.mse_loss(s1_qvalue, target_reward_tensor, reduction='mean')
                    loss.backward()
                    optimizer.step()

                self.is_new_episode = False
                if self.game.is_episode_finished():
                    break

        self.game.close()

    def test(self):
        pass

    def _preprocess_frame(self, frame):
        # Crop the screen (remove the roof because it contains no information)
        cropped_frame = frame[30:-10, 30:-30]
        # Normalize Pixel Values
        normalized_frame = cropped_frame / 255.0
        # Resize
        preprocessed_frame = transform.resize(normalized_frame, self.input_size)
        return preprocessed_frame

    def _stack_frames(self, img):
        img_preprocessed = self._preprocess_frame(rgb2gray(img.transpose(1, 2, 0)))
        if self.is_new_episode:
            for _ in range(self.stack_size):
                self.stacked_frames.append(img_preprocessed)
        else:
            self.stacked_frames.append(img_preprocessed)
        stacked_frames_numpy = np.stack(self.stacked_frames, axis=0)
        return stacked_frames_numpy

    def _get_epsilon_greedy_action(self, state_1, epsilon):
        z = random.uniform(0, 1)
        if z <= epsilon:
            action_1 = np.random.choice(self.action_size)
        else:
            self.model.eval()
            action_1 = torch.argmax((self.model(state_1))).item()
            self.model.train()
        return action_1


if __name__ == "__main__":
    dqn = DQNetwork()
    dqn.train()
    print(dqn)
