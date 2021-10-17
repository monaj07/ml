"""
Twin Delayed DDPG (TD3)
This algorithm is an extension of DDPG, which makes it more stable,
by making multiple minor changes compared to DDPG.

First of all, it introduces a twin version of the Q-network (critic),
and keeps updating both critics with parameters φ1 and φ2.
Then, for setting the target value y for the critic network,
(which was y = r + gamma * (1 - done) * Q'[s2, a'|φ'], a'=µ(s2|θ'), in DDPG),
it uses the minimum value of the two network,
i.e. Q'_min = min{Q'[s2, a'|φ'_1], Q'[s2, a'|φ'_2]},
and y = r + gamma * (1 - done) * Q'_min.

Another change is introduced by adding noise and smoothing the action a' taken in s2.
Previously in DDPG we used a'=µ(s2|θ') for computing target value for critic,
but now we add some variability to it and then smooth it:
a'=clip(µ(s2|θ') + noise, a_min, a_max), noise = clip(ε, -c, c), ε ~ Normal

Last but not least, the policy network and its target version (µ(s|θ) and µ(s|θ')),
as well as the two target versions of the critic (Q(s, a|φ'_1) and Q(s, a|φ'_2))
are updated with a smaller rate, compared to the critic network updates.

Note:
For updating policy, we can simply use Q(s, a|φ_1) in the maximum likelihood part.

* Training procedure:
https://spinningup.openai.com/en/latest/algorithms/td3.html

Randomly initialize critic networks {Q(s, a|φ_1), Q(s, a|φ_2)} and actor µ(s|θ), with weights θ, φ_1, φ_2
Empty experience replay buffer D
Initialize target networks with weights θ' <- θ and φ'_1 <- φ_1, φ'_2 <- φ_2
REPEAT:
    Observe state s and select action a = clip(µ(s|θ) + ε, a_min, a_max), ε ~ Normal
    (Remember that this action is computed in no_grad() mode and eval() mode)
    Execute a
    Record reward r, next state s2, and whether the episode is terminated (done)
    Store (s, a, r, s2, done) in buffer D
    Restart the episode if s2 is done, i.e. Reset the environment
    If it is time to update φ_1, φ_2:
        Randomly sample a batch B={(s, a, r, s2, done)} from D
        Compute target actions:
            a'=clip(µ(s2|θ') + noise, a_min, a_max), noise = clip(ε, -c, c), ε ~ Normal
        Compute targets values for the critic network (while keeping actor network fixed):
            y = r + gamma * (1 - done) * min{Q'[s2, a'|φ'_1], Q'[s2, a'|φ'_2]}
        Update critic networks by one-step gradient descent:
            MSE Loss(φ_1):  {|Q(s, a|φ_1) - y|^2}  (over the batch)
            MSE Loss(φ_2):  {|Q(s, a|φ_2) - y|^2}  (over the batch)
            (More explanation in compute_expected_q_s1_a1_for_critic_network method docstring)
        If it is time to update θ, θ', φ'_1, φ'_2:
            Update policy network by one-step gradient ascent:
                Maximum Likelihood: J(θ) = {Q(s, µ(s|θ)|φ_1)}  (over the batch)
                Gradient to be propagated through two networks: ∇_θ.J(θ) ≈ ∇_a.Q(s, a|φ_1) * ∇_θ.µ(s|θ)
                (More explanation in compute_expected_q_s1_a1_for_actor_network method docstring)
            Update the target network parameters {θ', φ'_1, φ'_2} using polyac averaging:
                φ'_1 <- ρφ'_1 + (1-ρ)φ_1
                φ'_2 <- ρφ'_2 + (1-ρ)φ_2
                θ' <- ρθ' + (1-ρ)θ

"""
from collections import deque
import numpy as np
import os
from os.path import dirname, abspath
import sys
import torch
from torch.distributions.normal import Normal
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(dirname(dirname(abspath(__file__))))
from agents.ddpg import DDPG
from networks.networks import ActorDDPG, CriticDDPG
from utilities import make_deterministic, OU_Noise


class TD3(DDPG):
    def __init__(self, input_dim, action_dimension,
                 set_device=None, gradient_clipping_norm=None,
                 learning_rate_actor=0.01, learning_rate_critic=0.01,
                 actor_noise_scale=0.1, steps_between_learning_steps=1,
                 max_episode_length=2000, polyac=0.99, seed=1364,
                 update_policy_and_targets_skip_rate=2,
                 target_actor_noise_scale=0.2, target_actor_noise_clip=0.5,
                 starting_iteration_to_follow_policy=0):
        super().__init__(input_dim, action_dimension,
                         set_device=set_device,
                         gradient_clipping_norm=gradient_clipping_norm,
                         learning_rate_actor=learning_rate_actor,
                         learning_rate_critic=learning_rate_critic,
                         actor_noise_scale=actor_noise_scale,
                         steps_between_learning_steps=steps_between_learning_steps,
                         max_episode_length=max_episode_length,
                         polyac=polyac, seed=seed)

        self.update_policy_and_targets_skip_rate = update_policy_and_targets_skip_rate
        self.target_actor_noise_scale = target_actor_noise_scale
        self.target_actor_noise_clip = target_actor_noise_clip
        self.starting_iteration_to_follow_policy = starting_iteration_to_follow_policy

        self.critic_net_twin = CriticDDPG(input_dim, action_dimension).to(self.device)
        self.critic_net_twin_target = CriticDDPG(input_dim, action_dimension).to(self.device)

        self.critic_net_twin_target.load_state_dict(self.critic_net_twin.state_dict())
        self.critic_net_twin_target.eval()

        # self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.optimizer_critic_twin = optim.Adam(self.critic_net_twin.parameters(),
                                                lr=self.learning_rate_critic, eps=1e-4)

    def compute_expected_q_s1_a1_for_critic_network(self, minibatch_state_1, minibatch_action_1):
        # In order to minimize the TD error, we need to optimize the parameters of the critic network.
        # The TD error will be the squared difference between the expected q_values
        # (given the current state and taken action in the experience) and the target q_values.
        q_a_state_1 = self.critic_net(
            minibatch_state_1.to(self.device),
            minibatch_action_1.to(torch.float).to(self.device)
        )
        q_a_state_1_twin = self.critic_net_twin(
            minibatch_state_1.to(self.device),
            minibatch_action_1.to(torch.float).to(self.device)
        )
        return q_a_state_1, q_a_state_1_twin

    def compute_target_q_state_1(self, minibatch_reward_1, minibatch_state_2, minibatch_finished, env):
        # Calculate the target rewards: R = r + gamma * Q_min(s2, µ(s2;θ'); φ'), OR R = r
        # Depending on whether we are in terminal state or not.
        with torch.no_grad():
            # Compute noisy target action for state 2
            # -----------------------------------------
            # a'=clip(µ(s2|θ') + noise, a_min, a_max), noise = clip(ε, -c, c), ε ~ Normal
            action_in_state_2_deterministic = self.actor_net_target(minibatch_state_2.to(self.device))
            target_action_noise = torch.randn_like(action_in_state_2_deterministic) * self.target_actor_noise_scale
            target_action_noise = torch.clip(
                target_action_noise,
                -self.target_actor_noise_clip,
                self.target_actor_noise_clip
            )
            action_in_state_2 = torch.clip(
                action_in_state_2_deterministic + target_action_noise,
                env.env.action_space.low.item(),
                env.env.action_space.high.item()
            )
            # -----------------------------------------

            q_state_2 = self.critic_net_target(
                minibatch_state_2.to(self.device),
                action_in_state_2
            ).squeeze(1).detach()
            q_state_2_twin = self.critic_net_twin_target(
                minibatch_state_2.to(self.device),
                action_in_state_2
            ).squeeze(1).detach()
            q_state_2_min = (
                    q_state_2 * (q_state_2 < q_state_2_twin) +
                    q_state_2_twin * (q_state_2 > q_state_2_twin)
            )
        q_state_1_target = minibatch_reward_1.to(self.device) + self.gamma * (
                (1 - minibatch_finished).to(self.device) * q_state_2_min
        )
        return q_state_1_target

    def optimise_critic(self, q_a_state_1, q_a_state_1_twin, q_state_1_target):
        # Optimise the network parameters based on TD loss (q_expected, q_target)
        loss1 = super().optimise_critic(q_a_state_1, q_state_1_target)
        # --------------------------
        self.optimizer_critic_twin.zero_grad()
        # loss = F.smooth_l1_loss(q_a_state_1_twin, q_state_1_target.view(-1, 1))
        loss2 = F.mse_loss(q_a_state_1_twin, q_state_1_target.view(-1, 1))
        loss2.backward()
        if self.gradient_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic_net_twin.parameters(),
                                           self.gradient_clipping_norm)
        self.optimizer_critic_twin.step()
        return loss1, loss2

    def learning_step(self, minibatch, env, update_policy_and_targets=True):
        # Compute loss over the minibatch and optimise the network parameters
        minibatch_state_1 = minibatch[0]
        minibatch_action_1 = minibatch[1]
        minibatch_reward_1 = minibatch[2]
        minibatch_state_2 = minibatch[3]
        minibatch_finished = minibatch[4]

        # Learn the critic:
        # -----------------------------------
        # Compute Q(s1, a1)
        q_a_state_1, q_a_state_1_twin = self.compute_expected_q_s1_a1_for_critic_network(
            minibatch_state_1,
            minibatch_action_1
        )
        # Compute Q_target(s1, a1)
        q_state_1_target = self.compute_target_q_state_1(
            minibatch_reward_1,
            minibatch_state_2,
            minibatch_finished,
            env
        )
        # TD Optimisation:
        loss1, loss2 = self.optimise_critic(q_a_state_1, q_a_state_1_twin, q_state_1_target)
        # -----------------------------------

        if update_policy_and_targets:
            # Learn the actor:
            # -----------------------------------
            # Compute Q(s1, µ(s1))
            q_a_state_1 = self.compute_expected_q_s1_a1_for_actor_network(
                minibatch_state_1
            )
            # Maximum likelihood
            loss3 = self.optimise_actor(q_a_state_1)
            # -----------------------------------
            return loss1, loss2, loss3
        return loss1, loss2

    def run_single_episode(self, env, episode, number_of_learning_iterations_in_one_step=1):
        # Make each episode deterministic based on the total_iteration_number
        make_deterministic(self.total_steps_so_far, env.env)

        finished = False
        episode_rewards = []
        episode_losses = []

        # Create the first state of the episode
        state_1 = env.get_state(episode_start=True)

        while not finished:
            env.env.render(mode='rgb_array')

            # Before getting to iteration `starting_iteration_to_follow_policy`,
            # perform a uniform action selection for a good exploration:
            if self.total_steps_so_far < self.starting_iteration_to_follow_policy:
                action_1 = torch.from_numpy(env.env.action_space.sample()).unsqueeze(0).float()
            else:
                action_1 = self.get_action(env, state_1)
            # Take the selected action in the environment
            s2, reward_1, finished, _ = env.env.step(action_1)

            # when episode is finished, state_2 does not matter,
            # and won't contribute to the optimisation
            # (because state_1 was the last state of the episode)
            state_2 = (0 * state_1) if finished else env.get_state()

            # Add the current transition (s, a, r, s', done) to the replay memory
            self.add_experience_to_replay_memory(
                state_1,
                action_1,
                reward_1,
                state_2,
                finished
            )

            # Policy Network optimisation:
            # ----------------------------
            # If there are enough sample transitions inside the replay_memory,
            # then we can start training our policy network using them;
            # Otherwise we move on to the next state of the episode.
            if len(self.replay_memory) >= self.batch_size:
                if self.total_steps_so_far % self.steps_between_learning_steps == 0:
                    for internal_training_step in range(number_of_learning_iterations_in_one_step):
                        # Take a random sample minibatch from the replay memory
                        minibatch = self.sample_from_replay_memory(self.batch_size)

                        # Compute the TD loss over the minibatch
                        update_policy_and_targets = ((self.total_steps_so_far + internal_training_step) %
                                                     self.update_policy_and_targets_skip_rate) == 0
                        _ = self.learning_step(minibatch, env, update_policy_and_targets)

                        # Track the value of loss (for debugging purpose)
                        # episode_losses.append(loss.item())

                        # Update the target networks (polyac averaging)
                        if update_policy_and_targets:
                            self.soft_update_target_networks()

            # Go to the next step of the episode
            state_1 = state_2
            # Add up the rewards collected during this episode
            episode_rewards.append(reward_1)
            # One single training iteration is passed
            self.total_steps_so_far += 1

            # If the agent has received a satisfactory episode reward, stop it.
            if sum(episode_rewards) >= env.score_required_to_win:
                finished = True

            # If the episode takes longer than 'max_episode_length', terminate it.
            if len(episode_rewards) > self.max_episode_length:
                break

            # print(f"episode: {episode}, reward: {reward_1}, action_1: {action_1}")

        # Return the total rewards collected within this single episode run
        return episode_rewards

    def save_snapshot(self, episode, reward):
        try:
            state = {
                'episode': episode,
                'actor_state_dict': self.actor_net.state_dict(),
                'actor_optimizer': self.optimizer_actor.state_dict(),
                'critic_state_dict': self.critic_net.state_dict(),
                'critic_optimizer': self.optimizer_critic.state_dict(),
                'critic_twin_state_dict': self.critic_net_twin.state_dict(),
                'critic_twin_optimizer': self.optimizer_critic_twin.state_dict()
            }
            os.makedirs('./model_snapshots', exist_ok=True)
            torch.save(state, os.path.join('./model_snapshots', f'snapshot_episode_{episode}_reward_{reward}.pth'))
        except Exception as e:
            print(f"Unable to save the model because:\n {e}")

    def soft_update_target_networks(self, polyac=None):
        if polyac is None:
            polyac = self.polyac
        super().soft_update_target_networks(polyac)
        for p, p_target in zip(self.critic_net_twin.parameters(), self.critic_net_twin_target.parameters()):
            p_target.data.copy_(polyac * p_target.data + (1 - polyac) * p.data)


if __name__ == "__main__":
    seed = 1364
    td3_learner = TD3((40, 90), 1, seed=seed)
    print(td3_learner)
