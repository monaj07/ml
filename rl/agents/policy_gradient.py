"""
Policy Gradient algorithm
In Policy Gradient, we directly model the policy pi_theta(a|s).
For each state, this function outputs a probability distribution over all possible actions.

The training is implemented this way:

Starting from the beginning of the episode, we start off with a randomly initialised pi(a|s),
and given s0, we calculate pi(a0|s0) and take a sample action (a0) from its output probability distribution.
(Here we use Categorical module of pytorch to perform sampling).
We also need to record the log_prob of that action (i.e. log(pi(a0|s0))),
to use it in the weighted maximum likelihood optimisation. (weights are return values G_t)
(Another nice feature of Categorical module is that we can simply get that log_prob for that action.

    def get_action_and_log_prob(self, state_1):
        policy_action_probs = self.policy_net(state_1.to(self.device))
        policy_action_distribution = Categorical(logits=policy_action_probs)
        action = policy_action_distribution.sample()
        log_prob = policy_action_distribution.log_prob(action)
        return action.item(), log_prob
We then take that sampled action to move to the next state of the environment: env.step(a0) -> s1
and record the achieved reward in that step.

We then repeat the above process by calling 'get_action_and_log_prob(s1)' again,
and for every single step in the episode, we record {log_prob(a_t), r_t} (two scalar values).
Once the episode is finished, we need to compute the return values (G_t) for each recorded step in the episode.
On top of that, we need to apply discount factor as well.
The return is computed from the recorded episode reward vector this way:
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + (discount ^ 2) * x2,
         x1 + discount * x2,
         x2]
This gives the (reward-to-go) G_t for every single step of the episode.
If you don't want to use reward-to-go option, just set it to False,
then it will assign the same return value
(which is the return at the beginning of the episode G_0) to all steps of the episode.

At the end of the episode, we have two vectors of [log_probs] and [G_t].
We can do weighted maximum log-likelihood (each sample is the log_prob(a_t) weighted by its G_t).
However it is recommended to run this in large batches, hence we fill our
self.episode_return_batch and self.log_probs_batch, until it reaches the batch size.
Then we compute the loss (which is mean of negative of weighted log-likelihood), and back propagate.

It is importance to know that once this single back-propagation is done,
our policy function, pi_theta(a|s), has been updated, and as a result,
the previous samples that were recorded based on the old policy function are useless for training.
We need to generate new samples from the updated policy.
Therefore we go back to line 8 of this docstring (above) and repeat the process,
with the difference that pi_theta(a|s) is NOT randomly initialised anymore, it has been updated :).
"""
from collections import deque
import numpy as np
import os
from os.path import dirname, abspath
import sys
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(dirname(dirname(abspath(__file__))))
from exploration.exploration_strategies import ActionExplorer
from networks.network_builder import CreateNet
from utilities import make_deterministic


class VanillaPolicyGradient:
    """
    """
    def __init__(self, input_dim, num_actions, network_params=None,
                 set_device=None, gradient_clipping_norm=None,
                 reward_to_go=True,
                 learning_rate=0.01, seed=1364):
        self.seed = seed
        # Training parameters
        self.gamma = 0.99
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

        # Policy gradient parameters
        self.episode_return_batch = torch.Tensor()
        self.actions_batch = torch.Tensor()
        self.log_probs_batch = torch.Tensor()
        self.reward_to_go = reward_to_go
        self.batch_size = 4096  # Batch size for Policy Gradient should be large to reduce variance

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

        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, eps=1e-4)

    def optimise(self):
        self.optimizer.zero_grad()
        # Shuffle the minibatch
        permutation = torch.randperm(len(self.episode_return_batch))
        self.episode_return_batch = self.episode_return_batch[permutation]
        self.log_probs_batch = self.log_probs_batch[permutation]
        # Negative log-likelihood loss
        loss = -(self.log_probs_batch * self.episode_return_batch).mean()
        loss.backward()
        if self.gradient_clipping_norm is not None:
            # clip gradients to help stabilise training
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                           self.gradient_clipping_norm)

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

    def learning_step(self):
        loss = self.optimise()
        # Once the recorded minibatch rewards and policy probabilities are used for one optimisation step,
        # throw them away, as they are sampled from the out-dated policy.
        # Now that the policy has been updated, new set of samples should be recorded.
        self.episode_return_batch = torch.Tensor()
        self.log_probs_batch = torch.Tensor()
        return loss

    def get_action_and_log_prob(self, state_1):
        policy_action_probs = self.policy_net(state_1.to(self.device))
        policy_action_distribution = Categorical(logits=policy_action_probs)
        action = policy_action_distribution.sample()
        log_prob = policy_action_distribution.log_prob(action)
        return action.item(), log_prob

    def run_single_episode(self, env, episode=None):
        # Make each episode deterministic based on the total_iteration_number
        make_deterministic(self.total_steps_so_far, env.env)

        finished = False
        episode_rewards = []
        log_probs = []

        # Create the first state of the episode
        state_1 = env.get_state(episode_start=True)

        while not finished:
            action, log_prob = self.get_action_and_log_prob(state_1)
            # Take the selected action in the environment
            _, reward, finished, _ = env.env.step(action)

            episode_rewards.append(reward)
            log_probs.append(log_prob)

            # If not finished, set the current_state as previous state
            if not finished:
                state_1 = env.get_state()

            # One single training iteration is passed
            self.total_steps_so_far += 1

            # If the agent has received a satisfactory episode reward, stop it.
            if sum(episode_rewards) >= env.score_required_to_win:
                finished = True

        ep_len = len(episode_rewards)

        # Computing episode return for all time points in the episode (G_t)
        # input: vector [x0, x1, x2], output: [x0 + discount * x1 + (discount ^ 2) * x2,
        #                                      x1 + discount * x2,
        #                                      x2]
        if self.reward_to_go:
            episode_return = torch.tensor([
                    sum(episode_rewards[i:]*(self.gamma ** np.arange(ep_len-i)))
                    for i in range(ep_len)
            ])
        else:
            episode_return = torch.ones(ep_len) * sum(episode_rewards)

        self.episode_return_batch = torch.cat([self.episode_return_batch, episode_return])
        self.log_probs_batch = torch.cat([self.log_probs_batch, torch.cat(log_probs)])

        # Policy Network optimisation:
        # ----------------------------
        if len(self.episode_return_batch) >= self.batch_size:
            _ = self.learning_step()

        # Return the total rewards collected within this single episode run
        return episode_rewards

    def save_snapshot(self, episode, reward):
        try:
            state = {
                'episode': episode,
                'state_dict': self.policy_net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            os.makedirs('./model_snapshots', exist_ok=True)
            torch.save(state, os.path.join('./model_snapshots',
                                           f'snapshot_{self.__class__.__name__}_episode_{episode}_reward_{reward}.pth'))
        except Exception as e:
            print(f"Unable to save the model because:\n {e}")


if __name__ == "__main__":
    seed = 1364
    set_device = 'cpu'
    learning_rate = 0.001
    network_params = {
        'input_dim': 4,
        'dense_layers': [30, 15, 2],
        'activation': 'relu',
        'dense_bn': True
    }
    agent = VanillaPolicyGradient(4, 2,
                network_params=network_params,
                set_device=set_device,
                learning_rate=learning_rate,
                seed=seed)
    print()
