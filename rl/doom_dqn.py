from collections import deque
import copy
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from skimage import transform
from skimage.color import rgb2gray
import torch
from torch import nn
import torch.nn.functional as F
import time
from vizdoom import *

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

logging.basicConfig(filename=f"{datetime.now().strftime('%Y-%m-%d--%H-%M')}.log", format='%(message)s', level=logging.INFO)


def create_doom_env():
    game = DoomGame()
    game.load_config(
        "../../../AppData/Local/Continuum/miniconda3/envs/py372/Lib/site-packages/vizdoom/scenarios/basic.cfg")
    game.set_doom_scenario_path(
        "../../../AppData/Local/Continuum/miniconda3/envs/py372/Lib/site-packages/vizdoom/scenarios/basic.wad")
    game.init()

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

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
        self.dropout = nn.Dropout(0.5)
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
        x4 = self.dropout(F.elu(self.fc_1(x3)))
        # output dense layer
        x5 = self.fc_2(x4)
        return x5


class DQNetwork:
    def __init__(self):
        # Exploration parameters for epsilon greedy strategy
        self.epsilon_start = .95  # exploration probability at start
        self.epsilon_stop = 0.01  # minimum exploration probability
        self.decay_rate = 0.0001  # exponential decay rate for exploration prob

        # TRAINING HYPERPARAMETERS
        self.total_episodes = 1010  # Total episodes for training
        self.max_steps = 100  # Max possible steps in an episode
        self.batch_size = 128
        self.training = True
        self.is_new_episode = True

        # Q learning hyperparameters
        self.gamma = 0.99  # Discounting rate
        self.learning_rate = 0.0002  # Alpha (aka learning rate)

        # MEMORY HYPERPARAMETERS
        self.memory_size = 50000  # Number of experiences the Memory can keep
        self.experience_box = deque([], maxlen=self.memory_size)

        # Input properties
        self.stack_size = 4
        self.input_size = (84, 84)
        self.stacked_frames = deque([], maxlen=self.stack_size)

        # Initialising the game environment
        self.game, self.possible_actions = create_doom_env()
        self.action_size = self.game.get_available_buttons_size()

        # Set computing device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Building the Q-Network
        self.model = BuildQNet(self.action_size, self.stack_size)
        self.model.to(device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(device)
        self.target_update = 5

    def train(self):
        self.model.train()
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(),
        #     lr=self.learning_rate,
        #     momentum=0.99,
        #     weight_decay=0.0005
        # )
        optimizer = torch.optim.RMSprop(self.model.parameters())

        total_steps_so_far = 0

        for episode in range(self.total_episodes):
            # Start a new episode
            self.game.new_episode()

            # self.new_episode is used to check if we are at the start of the spisode,
            # in which case the 4-channel stacked_frames is instantiated, using four
            # copies of the first frame of the episode (in the _stack_frames method):
            self.is_new_episode = True

            epsilon = self.epsilon_stop
            episode_loss = []
            episode_rewards = []
            explored_actions = []
            actions_in_episode = [0, 0, 0]
            for _ in range(self.max_steps):
                # Set the value of epsilon for explore/exploit trade-off
                epsilon = self.epsilon_stop + (
                        (self.epsilon_start - self.epsilon_stop) *
                        np.exp(-self.decay_rate * total_steps_so_far)
                )

                # Append the transition (s, a, r, s') that is observed
                # in the episode to the experience_box.
                state = self.game.get_state()
                img = state.screen_buffer

                total_steps_so_far += 1

                # Generate a transition
                state_1 = self._stack_frames(img)
                state_1_tensor = torch.tensor(state_1, dtype=torch.float32).unsqueeze(0)
                action_1, explored = self._get_epsilon_greedy_action(state_1_tensor, epsilon, episode)
                actions_in_episode[action_1] += 1
                reward_1 = self.game.make_action(self.possible_actions[action_1])
                episode_rewards.append(reward_1)
                explored_actions.append(explored)

                # As soon as you make any action within the episode,
                # you will be at least in the 2nd step of the episode onward,
                # so the `is_new_episode` flag should be False
                # to append subsequent frames to the previous stack: state_1 -> state_2
                self.is_new_episode = False

                next_state_is_terminal = self.game.is_episode_finished()
                # next_state_is_terminal = (reward_1 == 100)

                if next_state_is_terminal:
                    # Does not matter what you put in state_2,
                    # since it is a terminal state
                    # and its content won't contribute into the target_reward
                    state_2 = 0 * state_1
                else:
                    state_2 = self._stack_frames(self.game.get_state().screen_buffer)

                # Pack the transition into a tuple
                transition = (state_1, action_1, reward_1, state_2, next_state_is_terminal)

                # Append the transition to the experience replay box
                if 1:  # next_state_is_terminal or ((total_steps_so_far % 4) == 0):
                    self.experience_box.append(transition)

                # Sample a random minibatch of transitions from experience_box,
                # every `q_update_freq` steps
                q_update_freq = 2
                if (total_steps_so_far % q_update_freq) == 0:
                    if len(self.experience_box) >= 4 * self.batch_size:
                        # Reset the gradients
                        optimizer.zero_grad()

                        # Create a minibatch
                        experience_numpy = np.array(self.experience_box)
                        minibatch_indices = np.random.choice(experience_numpy.shape[0], self.batch_size, replace=False)
                        minibatch = experience_numpy[minibatch_indices, :]

                        # Calculate the predictions for Q(s1, a1; theta)
                        s1_tensor = torch.tensor(np.stack(minibatch[:, 0], axis=0), dtype=torch.float32)
                        s1_qvalue = self.model(s1_tensor.to(self.device))

                        # Calculate the target rewards: R = r + gamma * max_a2{Q(s2, a2; theta)}, OR R = r
                        # Depending on whether we are in terminal state or not
                        target_reward_tensor = torch.tensor(np.stack(minibatch[:, 2], axis=0)).to(self.device)
                        s2_is_terminal_tensor = torch.tensor(np.stack(minibatch[:, 4], axis=0)).to(self.device)
                        s2_tensor = torch.tensor(np.stack(minibatch[:, 3], axis=0), dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            self.target_model.eval()
                            s2_max_qvalue = self.target_model(s2_tensor).max(dim=-1)[0].detach()
                        target_reward_tensor = target_reward_tensor + (
                                self.gamma * (~s2_is_terminal_tensor) * s2_max_qvalue
                        )

                        loss = F.smooth_l1_loss(s1_qvalue, target_reward_tensor.float().view(-1, 1))
                        # loss = F.mse_loss(s1_qvalue, target_reward_tensor.float().view(-1, 1), reduction='mean')
                        loss.backward()
                        for param in self.model.parameters():
                            param.grad.data.clamp_(-1, 1)
                        optimizer.step()
                        episode_loss.append(loss.item())

                        # del experience_numpy

                if next_state_is_terminal or self.game.is_episode_finished():
                    break
            try:
                print(f'episode: {episode},\t'
                      f'num_steps: {len(episode_rewards)},\t'
                      f'epsilon: {round(epsilon, 2)},\t'
                      f'explored_actions: {round(100 * sum(explored_actions) / max(len(explored_actions), 1))}%,\t'
                      f'actions: {actions_in_episode},\t'
                      f'average_episode_loss = {round(sum(episode_loss) / float(max(len(explored_actions), 1)), 2)},\t'
                      f'total_reward = {sum(episode_rewards)}')
                logging.info(f'episode: {episode},\t'
                             f'num_steps: {len(episode_rewards)},\t'
                             f'epsilon: {round(epsilon, 2)},\t'
                             f'explored_actions: {round(100 * sum(explored_actions) / max(len(explored_actions), 1))}%,\t'
                             f'average_episode_loss = {round(sum(episode_loss) / float(max(len(explored_actions), 2)), 1)},\t'
                             f'total_reward = {sum(episode_rewards)}')
            except Exception as e:
                print(f'exception:\n{e}')

            if (episode % 100) == 0 and episode > 0:
                try:
                    print('taking snapshot from the trained model so far...')
                    state = {
                        'epoch': episode,
                        'state_dict': self.model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    os.makedirs('doom_dqn_models', exist_ok=True)
                    torch.save(state, os.path.join('doom_dqn_models', f'snapshot_episode_{episode}.pth'))
                except Exception as e:
                    print(f"Unable to save the model because:\n {e}")

            # Update the target network, copying all weights and biases in DQN
            if (episode % self.target_update) == 0:
                self.target_model.load_state_dict(self.model.state_dict())

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

    def _get_epsilon_greedy_action(self, state_1, epsilon, episode):
        z = random.uniform(0, 1)
        if z <= epsilon:
            explored = 1
            action_1 = np.random.choice(self.action_size)
        else:
            explored = 0
            with torch.no_grad():
                self.model.eval()
                action_1 = torch.argmax((self.model(state_1.to(self.device)))).item()
                self.model.train()
        return action_1, explored


if __name__ == "__main__":
    dqn = DQNetwork()
    dqn.train()
    print(dqn)
