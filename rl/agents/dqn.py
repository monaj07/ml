"""
DQN algorithm
-------------
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm

To minimise the error, we will use the `Huber
loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
like the mean squared error when the error is small, but like the mean
absolute error when the error is large - this makes it more robust to
outliers when the estimates of :math:`Q` are very noisy.
"""
from collections import deque
import numpy as np
import os
from os.path import dirname, abspath
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(dirname(dirname(abspath(__file__))))
from networks.network_builder import CreateNet
from utilities import make_deterministic


class DQN:
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
    def __init__(self, input_dim, num_actions, network_params=None,
                 set_device=None, gradient_clipping_norm=None,
                 learning_rate=0.01, double_dqn=False, seed=1364):
        # Training parameters
        self.gamma = 0.99
        self.batch_size = 256
        self.target_network_update = 10
        self.total_steps_so_far = 0
        self.save_model_frequency = 100
        self.learning_rate = learning_rate
        self.latest_learning_rate = learning_rate
        self.gradient_clipping_norm = gradient_clipping_norm

        # Experience Replay Memory
        self.memory_size = 40000
        self.replay_memory = deque([], maxlen=self.memory_size)

        # ----------------------------------------
        # Make the algorithm outputs reproducible
        make_deterministic(seed)
        # ----------------------------------------

        # if gpu is to be used
        if set_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = torch.device(set_device)

        # Alternative DQN training
        self.double_dqn = double_dqn

        # Q-network instantiation
        # (Explanation of the network_params in networks/network_builder.py)
        if network_params is None:
            network_params = {
                'input_dim': input_dim,
                'conv_layers': [(3, 16, 5, 2), (16, 32, 5, 2), (32, 32, 5, 2)],
                'dense_layers': [num_actions],
                'conv_bn': True,
                'activation': 'relu'
            }
        self.policy_net = CreateNet(network_params).to(self.device)
        self.target_net = CreateNet(network_params).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, eps=1e-4)

    def sample_from_replay_memory(self, batch_size=None):
        sample_size = self.batch_size if batch_size is None else self.batch_size
        minibatch_indices = np.random.choice(len(self.replay_memory), sample_size, replace=False)
        minibatch_state_1 = torch.cat([self.replay_memory[idx][0] for idx in minibatch_indices])
        minibatch_action_1 = torch.cat([self.replay_memory[idx][1] for idx in minibatch_indices])
        minibatch_reward_1 = torch.cat([self.replay_memory[idx][2] for idx in minibatch_indices])
        minibatch_state_2 = torch.cat([self.replay_memory[idx][3] for idx in minibatch_indices])
        minibatch_finished = torch.cat([self.replay_memory[idx][4] for idx in minibatch_indices])
        return minibatch_state_1, minibatch_action_1, minibatch_reward_1, minibatch_state_2, minibatch_finished

    def add_experience_to_replay_memory(self, *experience):
        state_1 = experience[0]
        action_1 = experience[1]
        reward_1 = experience[2]
        state_2 = experience[3]
        finished = experience[4]
        action_1 = torch.tensor(action_1, dtype=torch.float32).unsqueeze(0)
        reward_1 = torch.tensor(reward_1, dtype=torch.float32).unsqueeze(0)
        finished = torch.tensor(finished, dtype=torch.float32).unsqueeze(0)
        # Pack the transition and add it to the replay memory
        transition = (state_1, action_1, reward_1, state_2, finished)
        self.replay_memory.append(transition)

    def compute_expected_q_s1_a1(self, minibatch_state_1, minibatch_action_1):
        # Note: Remember that you should compute Q(s1, a1)
        # (for actions that you have taken in minibatch_actions_1,
        # not for all actions, that is why we need to call "gather" method here
        # to get the value only for those actions)
        q_a_state_1 = self.policy_net(minibatch_state_1.to(self.device)).gather(
            1, minibatch_action_1.to(torch.int64).unsqueeze(1).to(self.device)
        )
        return q_a_state_1

    def compute_target_q_state_1(self, minibatch_reward_1, minibatch_state_2, minibatch_finished):
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
        return q_state_1_target

    def optimise(self, q_a_state_1, q_state_1_target):
        # Optimise the network parameters based on TD loss (q_expected, q_target)
        self.optimizer.zero_grad()
        # loss = F.smooth_l1_loss(q_a_state_1, q_state_1_target.view(-1, 1))
        loss = F.mse_loss(q_a_state_1, q_state_1_target.view(-1, 1))
        loss.backward()
        if self.gradient_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                           self.gradient_clipping_norm) #clip gradients to help stabilise training

        self.optimizer.step()
        return loss

    def update_learning_rate(self, rolling_results, score_required_to_win):
        """
        Lowers the learning rate according to how close we are to the solution.
        (Function burrowed from
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/Base_Agent.py)
        """
        starting_lr = self.learning_rate
        last_rolling_score = rolling_results[-1]
        if last_rolling_score > 0.95 * score_required_to_win:
            new_lr = starting_lr / 1000.0
        elif last_rolling_score > 0.9 * score_required_to_win:
            new_lr = starting_lr / 500.0
        elif last_rolling_score > 0.75 * score_required_to_win:
            new_lr = starting_lr / 100.0
        elif last_rolling_score > 0.6 * score_required_to_win:
            new_lr = starting_lr / 20.0
        elif last_rolling_score > 0.5 * score_required_to_win:
            new_lr = starting_lr / 10.0
        elif last_rolling_score > 0.25 * score_required_to_win:
            new_lr = starting_lr / 2.0
        else:
            return -1

        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        return new_lr

    def learning_step(self, minibatch):
        # Compute loss over the minibatch and optimise the network parameters
        minibatch_state_1 = minibatch[0]
        minibatch_action_1 = minibatch[1]
        minibatch_reward_1 = minibatch[2]
        minibatch_state_2 = minibatch[3]
        minibatch_finished = minibatch[4]

        # Compute Q(s1, a1)
        q_a_state_1 = self.compute_expected_q_s1_a1(
            minibatch_state_1,
            minibatch_action_1
        )

        # Compute Q_target(s1, a1)
        q_state_1_target = self.compute_target_q_state_1(
            minibatch_reward_1,
            minibatch_state_2,
            minibatch_finished
        )

        # Optimisation:
        loss = self.optimise(q_a_state_1, q_state_1_target)
        return loss

    def save_snapshot(self, episode, reward):
        try:
            state = {
                'episode': episode,
                'state_dict': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            os.makedirs('./model_snapshots', exist_ok=True)
            torch.save(state, os.path.join('./model_snapshots', f'snapshot_episode_{episode}_reward_{reward}.pth'))
        except Exception as e:
            print(f"Unable to save the model because:\n {e}")


if __name__ == "__main__":
    seed = 1364
    dqlearner = DQN((40, 90), 2, double_dqn=False, seed=seed)
