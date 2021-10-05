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
from agents.policy_gradient import VanillaPolicyGradient
from environments.cartpole_v0 import CartPoleV0Simple4D


def train(env, agent,
          total_episodes=5001,
          rolling_window_size=100,
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
    log_content = {**env.__dict__, **agent.__dict__}
    if "explorer" in agent.__dict__:
        log_content = {**log_content, **agent.explorer.__dict__}
    logging.info(log_content)
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

        # The following step
        # (dynamically reducing the learning rate based on the reward)
        # is very very important to converge to the right point
        if episode > 10:  # 10 is just an arbitrary number
            new_lr = agent.update_learning_rate(rolling_results, env.score_required_to_win)
            if new_lr > 0:
                logging.info(f" -- Dropped the learning rate to {new_lr}")
                sys.stdout.write(f" -- Dropped the learning rate to {new_lr}")
                sys.stdout.flush()

        # Do a complete single episode run
        episode_rewards = agent.run_single_episode(env, episode=episode)

        # Append the total rewards collected in the above finished episode
        episode_total_rewards.append(sum(episode_rewards))

        # Compute the rolling average reward (over the last 100 episodes)
        rolling_window = min([rolling_window_size, len(episode_total_rewards)])
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

        if (episode % reward_curve_display_frequency) == 0 and episode > 0:
            plot_durations(episode_total_rewards, rolling_results, log_tag)

        text = f"""Episode {episode}, """
        text += f""" Score: {episode_total_rewards[-1]:.2f}, """
        text += f""" Max score seen: {max_reward_so_far:.2f}, """
        if "explorer" in agent.__dict__:
            text += f""" Epsilon: {round(agent.explorer.current_epsilon, 3):.2f}, """
        if episode >= rolling_window_size:
            text += f""" Rolling score: {rolling_results[-1]:.2f}, """
            text += f""" Max rolling score seen: {max_rolling_score_seen:.2f}"""
        logging.info(text)
        sys.stdout.write("\r" + text)
        sys.stdout.flush()

        # When the agent has received enough reward, terminate the training
        if max_rolling_score_seen >= env.average_score_required_to_win:
            plot_durations(episode_total_rewards, rolling_results, log_tag)
            break


if __name__ == "__main__":
    # Negative seed means the algorithm runs in stochastic mode
    # In other words, seed=-1 leads to Non-producible outputs
    seed = 1
    rolling_window_size = 100
    reward_curve_display_frequency = 100
    save_model_frequency = 100

    env = CartPoleV0Simple4D(seed=seed)

    parameters = {
        'dqn': {
            'parameters': {
                'input_dim': env.input_dim,
                'num_actions': env.num_actions,
                'network_params': {
                    'input_dim': env.input_dim,
                    'dense_layers': [30, 15, env.num_actions],
                    'activation': 'relu',
                    'dense_bn': True
                },
                'explorer': ActionExplorer(epsilon_decay=0.005, seed=seed),
                'gradient_clipping_norm': 0.7,
                'set_device': 'cpu',
                'learning_rate': 0.01,
                'double_dqn': False,
                'seed': seed
            },
            'total_episodes': 501
        },
        'vpg': {
            'parameters': {
                'input_dim': env.input_dim,
                'num_actions': env.num_actions,
                'network_params': {
                    'input_dim': env.input_dim,
                    'dense_layers': [30, 15, env.num_actions],
                    'activation': 'relu',
                    # 'dense_bn': True
                },
                # 'gradient_clipping_norm': 0.7,
                'reward_to_go': True,
                'set_device': 'cpu',
                'learning_rate': 0.01,
                'seed': seed
            },
            'total_episodes': 10001
        }
    }

    agents = {
        'dqn': DQN(**parameters['dqn']['parameters']),
        'vpg': VanillaPolicyGradient(**parameters['vpg']['parameters'])
    }

    for agent_name in ['dqn']:
        agent = agents[agent_name]
        # Run training
        train(
            env,
            agent,
            total_episodes=parameters[agent_name]['total_episodes'],
            rolling_window_size=rolling_window_size,
            reward_curve_display_frequency=reward_curve_display_frequency,
            save_model_frequency=save_model_frequency
        )
