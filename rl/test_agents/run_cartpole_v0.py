"""
Run RL agents through CartPole-V0 environment.
"""
from datetime import datetime
import logging
import numpy as np
import os
from os.path import dirname, abspath
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
from utilities import plot_durations
from exploration.exploration_strategies import ActionExplorer
from agents.dqn import DQN
from environments.cartpole_v0 import CartPoleV0


def train(env, agent, explorer,
          total_episodes=5001,
          reward_curve_display_frequency=100,
          save_model_frequency=100
          ):
    """
    This function runs the agent through the environment 'evnv'
    for a number of 'total_episodes' episodes.
    During each single run episode, the agent learns to collect more rewards.
    """
    os.makedirs('./logs_and_figs', exist_ok=True)
    log_tag = datetime.now().strftime('%Y-%m-%d--%H-%M')
    logging.basicConfig(
        filename=f"./logs_and_figs/{log_tag}.log",
        format='%(message)s',
        level=logging.INFO
    )

    logging.info("Agent and Environment:")
    logging.info("-"*80)
    logging.info({**explorer.__dict__, **env.__dict__, **agent.__dict__})
    logging.info("-"*80)
    logging.info("\nTraining:\n")

    episode_total_rewards = []
    rolling_results = []
    max_reward_so_far = np.iinfo(np.int16).min
    max_rolling_score_seen = np.iinfo(np.int16).min

    # Run the game for total_episodes number
    for episode in range(total_episodes):
        # At the beginning of each episode, reset the environment
        env.env.reset()

        # Do a complete single episode run
        episode_rewards = env.run_single_episode(agent, explorer)

        # Append the total rewards collected in the above finished episode
        episode_total_rewards.append(sum(episode_rewards))

        # Compute the rolling average reward (over the last 100 episodes)
        rolling_window = min([100, len(episode_total_rewards)])
        rolling_results.append(np.mean(episode_total_rewards[-rolling_window:]))

        if episode_total_rewards[-1] > max_reward_so_far:
            max_reward_so_far = episode_total_rewards[-1]
            if len(episode_total_rewards) > save_model_frequency:
                # Save a snapshot of the best model so far
                agent.save_snapshot(episode, int(max_reward_so_far))

        if rolling_results[-1] > max_rolling_score_seen:
            max_rolling_score_seen = rolling_results[-1]

        if (episode % save_model_frequency) == 0 and episode > 0:
            # Save a snapshot at every "save_model_frequency" episode
            agent.save_snapshot(episode, int(sum(episode_rewards)))

        if (episode % agent.target_network_update) == 0:
            # Update the target network with the latest policy network parameters
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if (episode % reward_curve_display_frequency) == 0 and episode > 0:
            plot_durations(episode_total_rewards, rolling_results, log_tag)

        text = f"""Episode {episode}, """
        text += f""" Score: {episode_total_rewards[-1]:.2f}, """
        text += f""" Max score seen: {max_reward_so_far:.2f}, """
        text += f""" Epsilon: {round(explorer.current_epsilon, 3):.2f}"""
        if episode >= 100:
            text += f""" Rolling score: {rolling_results[-1]:.2f}, """
            text += f""" Max rolling score seen: {max_rolling_score_seen:.2f}"""
        logging.info(text)
        sys.stdout.write("\r" + text)
        sys.stdout.flush()


if __name__ == "__main__":
    seed = 1364
    total_episodes = 5001
    reward_curve_display_frequency = 100
    save_model_frequency = 100

    learning_rate = 0.001
    epsilon_decay = 0.0001
    gradient_clipping_norm = 0.7

    # Instantiate RL objects
    explorer = ActionExplorer(epsilon_decay=epsilon_decay, seed=seed)
    env = CartPoleV0(seed=seed)
    agent = DQN(env.input_dim, env.num_actions,
                gradient_clipping_norm=gradient_clipping_norm,
                learning_rate=learning_rate,
                double_dqn=True,
                seed=seed)
    # Run training
    train(
        env,
        agent,
        explorer,
        total_episodes=total_episodes,
        reward_curve_display_frequency=reward_curve_display_frequency,
        save_model_frequency=save_model_frequency
    )
