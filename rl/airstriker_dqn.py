"""
The code is working,
but it is not trained and tested due to its computational complexity
"""

from collections import deque
import copy
from datetime import datetime
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import retro

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), label='Single episode reward')
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='Average reward in the last 100 episodes')
    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQLearning:
    """
    If you are facing unstable training, the following ideas might help you.
    (From https://adgefficiency.com/dqn-solving/)
    ------------------------------------------
    1) Increasing the number of steps between target network updates (from 10 to something bigger) and,
    2) lowering the learning rate (from 0.01 to 0.001 or 0.0001),
    both reduce the speed of learning but should give more stability.
    In reinforcement learning stability is the killer -
    although sample efficiency is a big problem in modern reinforcement learning,
    you would rather have a slow, stable policy than a fast unstable one!
    Also,
    3) The idea behind increasing the size of the replay memory was (from 10,000 to 100,000)
    to smooth out the distribution of the batches.
    What I was quite often seeing was good performance up to the first 100,000 steps followed by collapse -
    so I thought that maybe the agent was suffering with changes in distribution over batches as it learnt.
    Cartpole doesn’t have a particularly complex state space,
    so it’s likely that all states are useful for learning throughout an agents lifetime.
    """
    def __init__(self):

        # Epsilon parameters
        self.epsilon_start = 0.99
        self.current_epsilon = self.epsilon_start
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.00001

        # Training parameters
        self.gamma = 0.99
        self.batch_size = 128
        self.target_network_update = 1
        self.total_steps_so_far = 0
        self.total_episodes = 501
        self.save_model_frequency = 50
        learning_rate = 0.00025

        # Experience Replay Memory
        self.memory_size = 1000000
        self.replay_memory = deque([], maxlen=self.memory_size)

        # Define the environment
        self.env = retro.make(game='Airstriker-Genesis')
        self.env.reset()

        # Get number of actions from gym action space
        self.num_actions = self.env.action_space.n
        self.possible_actions = np.array(np.identity(self.num_actions, dtype=int).tolist())

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get screen size so that we can initialize Q-network layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a cropped and down-scaled render buffer in get_screen()
        # the output of get_screen is a torch frame of shape (B, C, H, W)
        _, _, screen_height, screen_width = self.get_screen().shape

        # Q-network instantiation
        self.policy_net = DQN(screen_height, screen_width, self.num_actions).to(self.device)
        self.target_net = DQN(screen_height, screen_width, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.double_dqn = False

    def get_screen(self):
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([T.ToPILImage(),
                            T.Resize(110, interpolation=Image.CUBIC),
                            T.Grayscale(),
                            T.ToTensor()])

        # Resize, and add a batch dimension (BCHW)
        final_torch_screen = resize(screen).unsqueeze(0)
        return final_torch_screen

    def select_action(self, state, greedy=False):
        # -------------------------------------
        # Full-Greedy strategy (used at the test time)
        if greedy:
            with torch.no_grad():
                # take the greedy action
                action = self.policy_net(state.to(self.device)).max(-1)[1].item()
            return action
        # -------------------------------------
        # Epsilon-greedy strategy (mostly used in training time)
        self.current_epsilon = self.epsilon_end + (
                (self.epsilon_start - self.epsilon_end) *
                math.exp(-self.epsilon_decay * self.total_steps_so_far)
        )
        z = random.uniform(0, 1)
        if z < self.current_epsilon:
            # Take a random action
            action = random.choice(range(self.num_actions))
        else:
            with torch.no_grad():
                # take the greedy action
                action = self.policy_net(state.to(self.device)).max(-1)[1].item()
        return action
        # -------------------------------------

    def save_snapshot(self, episode, reward):
        try:
            state = {
                'episode': episode,
                'state_dict': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            os.makedirs('dqn_models', exist_ok=True)
            torch.save(state, os.path.join('dqn_models', f'snapshot_episode_{episode}_reward_{reward}.pth'))
        except Exception as e:
            print(f"Unable to save the model because:\n {e}")

    def train(self):
        logging.basicConfig(
            filename=f"{datetime.now().strftime('%Y-%m-%d--%H-%M')}.log",
            format='%(message)s',
            level=logging.INFO
        )
        episode_total_rewards = []
        max_reward_so_far = -1000
        max_reward_so_far_episode = 0
        for episode in range(self.total_episodes):
            self.env.reset()

            finished = False
            episode_rewards = []
            episode_losses = []

            # Create the first state of the episode
            prev_screen = self.get_screen()
            state_stack_1 = deque([], maxlen=4)
            for _ in range(4):
                state_stack_1.append(prev_screen)
            state_1 = torch.cat(list(state_stack_1), dim=1)

            while not finished:
                # select action and take that action in the environment
                action_1 = self.select_action(state_1)
                _, reward_1, finished, _ = self.env.step(self.possible_actions[action_1])
                episode_rewards.append(reward_1)

                # observe the next frame and create the next state
                if not finished:
                    state_stack_2 = copy.deepcopy(state_stack_1)
                    state_stack_2.append(self.get_screen())
                    state_2 = torch.cat(list(state_stack_2), dim=1)
                else:
                    # state_2 does not matter, since it won't contribute to the optimisation
                    # (it is the last state of the episode)
                    state_2 = 0 * state_1

                reward_1 = torch.tensor(reward_1, dtype=torch.float32).unsqueeze(0)
                action_1 = torch.tensor(action_1, dtype=torch.float32).unsqueeze(0)
                finished = torch.tensor(finished, dtype=torch.float32).unsqueeze(0)
                # Pack the transition and add it to the replay memory
                transition = (state_1, action_1, reward_1, state_2, finished)
                self.replay_memory.append(transition)

                # Go to the next step of the episode
                state_1 = state_2

                # Policy Network optimision:
                # If there are enough sample transitions inside the replay_memory,
                # then we can start training our policy network using them;
                # Otherwise we move on to the next state of the episode.
                if len(self.replay_memory) >= self.batch_size:
                    minibatch_indices = np.random.choice(len(self.replay_memory), self.batch_size, replace=False)
                    minibatch_state_1 = torch.cat([self.replay_memory[idx][0] for idx in minibatch_indices])
                    minibatch_action_1 = torch.cat([self.replay_memory[idx][1] for idx in minibatch_indices])
                    minibatch_reward_1 = torch.cat([self.replay_memory[idx][2] for idx in minibatch_indices])
                    minibatch_state_2 = torch.cat([self.replay_memory[idx][3] for idx in minibatch_indices])
                    minibatch_finished = torch.cat([self.replay_memory[idx][4] for idx in minibatch_indices])

                    # Compute Q(s1, a1)
                    # Note: Remember that you should compute Q(s, a)
                    # (for actions that you have taken in minibatch_actions_1,
                    # not for all actions, that is why we need to call "gather" method here
                    # to get the value only for those actions)
                    q_a_state_1 = self.policy_net(minibatch_state_1.to(self.device)).gather(
                        1, minibatch_action_1.to(torch.int64).unsqueeze(1).to(self.device)
                    )

                    # Calculate the target rewards: R = r + gamma * max_a2{Q'(s2, a2; theta)}, OR R = r
                    # Depending on whether we are in terminal state or not
                    # Note that for double-dqn, it would be r + gamma * Q'(s2, argmax{Q(s2, a2)})
                    with torch.no_grad():
                        if self.double_dqn:
                            action_2_max = self.policy_net(minibatch_state_2.to(self.device)).max(-1)[1].detach()
                            q_state_2_max = self.target_net(minibatch_state_2.to(self.device)).gather(
                                1, action_2_max.unsqueeze(1)
                            ).squeeze(1)
                        else:
                            q_state_2_max = self.target_net(minibatch_state_2.to(self.device)).max(-1)[0].detach()
                    q_state_1_target = minibatch_reward_1.to(self.device) + self.gamma * (
                        (1 - minibatch_finished).to(self.device) * q_state_2_max
                    )

                    # Optimisation:
                    self.optimizer.zero_grad()
                    loss = F.smooth_l1_loss(q_a_state_1, q_state_1_target.view(-1, 1))
                    episode_losses.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                    self.total_steps_so_far += 1

                if self.total_steps_so_far % 10 == 0:
                    plt.imshow(self.env.render(mode='rgb_array')), plt.show()

            episode_total_rewards.append(sum(episode_rewards))

            if sum(episode_rewards) > max_reward_so_far:
                max_reward_so_far = sum(episode_rewards)
                max_reward_so_far_episode = episode
                # Save a snapshot of the best model so far
                self.save_snapshot(episode, int(max_reward_so_far))

            if (episode % self.save_model_frequency) == 0:
                # Save a snapshot at every "save_model_frequency" episode
                self.save_snapshot(episode, int(sum(episode_rewards)))

            if (episode % self.target_network_update) == 0:
                # Update the target network with the latest policy network parameters
                self.target_net.load_state_dict(self.policy_net.state_dict())
                plot_durations(episode_total_rewards)
                print(f'episode: {episode},\t'
                      f'epsilon: {round(self.current_epsilon, 3)},\t'
                      f'max reward so far: {max_reward_so_far} in episode {max_reward_so_far_episode}')
                logging.info(f'episode: {episode},\t'
                             f'epsilon: {round(self.current_epsilon, 3)},\t'
                             f'max reward so far: {max_reward_so_far} in episode {max_reward_so_far_episode}')

    def test(self, model_filename, num_test_episodes=10):
        state_dict = torch.load(model_filename)['state_dict']
        self.policy_net.load_state_dict(state_dict)
        # self.policy_net.eval()
        self.policy_net.to(self.device)
        all_episodes_rewards = []
        for episode in range(num_test_episodes):
            self.env.reset()

            finished = False
            episode_rewards = []

            # Create the first state of the episode
            prev_screen = self.get_screen()
            state_stack_1 = deque([], maxlen=4)
            for _ in range(4):
                state_stack_1.append(prev_screen)
            state_1 = torch.cat(list(state_stack_1), dim=1)

            while not finished:
                # select action and take that action in the environment
                action_1 = self.select_action(state_1, greedy=True)
                _, reward_1, finished, _ = self.env.step(self.possible_actions[action_1])
                episode_rewards.append(reward_1)

                # observe the next frame and create the next state
                if not finished:
                    state_stack_1.append(self.get_screen())
                    state_1 = torch.cat(list(state_stack_1), dim=1)
                else:
                    break
            all_episodes_rewards.append(sum(episode_rewards))
        print(f'all_episodes_rewards: {all_episodes_rewards}')


if __name__ == "__main__":
    dqlearner = DQLearning()
    dqlearner.train()
    # dqlearner.test('dqn_models/snapshot_episode_300.pth', num_test_episodes=10)
