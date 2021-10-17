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
from agents.td3 import TD3
from environments.mountain_car_v0 import MountainCarV0


def train(env, agent,
          number_of_learning_iterations_in_one_step=1,
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
    agent_name = agent.__class__.__name__
    os.makedirs('./logs_and_figs', exist_ok=True)
    log_tag = datetime.now().strftime('%Y-%m-%d--%H-%M')
    logging.basicConfig(
        filename=f"./logs_and_figs/{log_tag}.log",
        format='%(message)s',
        level=logging.INFO
    )

    logging.info("\nAgent and Environment:\n")
    logging.info(f"{agent_name}")
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
        changed_lr_actor, changed_lr_critic = -1, -1
        if episode > 0:  # 0 is just an arbitrary number
            changed_lr_actor, changed_lr_critic = agent.update_learning_rate(
                rolling_results,
                env.score_required_to_win
            )

        # Do a complete single episode run
        episode_rewards = agent.run_single_episode(
            env,
            episode=episode,
            number_of_learning_iterations_in_one_step=number_of_learning_iterations_in_one_step
        )

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
            plot_durations(episode_total_rewards, rolling_results, log_tag, agent_name)

        text = f"""Episode {episode}, """
        text += f""" Score: {episode_total_rewards[-1]:.2f}, """
        text += f""" Max_score_seen: {max_reward_so_far:.2f}, """
        if "explorer" in agent.__dict__:
            text += f""" Epsilon: {round(agent.explorer.current_epsilon, 3):.2f}, """
        if changed_lr_actor > 0:
            text += f""" Changed_LR_Actor: {changed_lr_actor:.6f}, """
        if changed_lr_critic > 0:
            text += f""" Changed_LR_Critic: {changed_lr_critic:.6f}, """
        if episode >= 1:
            text += f""" Rolling_score: {rolling_results[-1]:.2f}, """
            text += f""" Max_rolling_score_seen: {max_rolling_score_seen:.2f}"""
        logging.info(text)
        sys.stdout.write("\r" + text)
        sys.stdout.flush()

        # When the agent has received enough reward, terminate the training
        if rolling_results[-1] >= env.average_score_required_to_win and episode > rolling_window_size / 2:
            plot_durations(episode_total_rewards, rolling_results, log_tag, agent_name)
            logging.info("-"*80)
            logging.info("Successfully passed the acceptable reward threshold.\n")
            sys.stdout.write("\nSuccessfully passed the acceptable reward threshold.\n")
            break


if __name__ == "__main__":
    # Negative seed means the algorithm runs in stochastic mode
    # In other words, seed=-1 leads to Non-reproducible outputs
    seed = 1
    number_of_learning_iterations_in_one_step = 10
    rolling_window_size = 100
    reward_curve_display_frequency = 10
    save_model_frequency = 100

    env = MountainCarV0(seed=seed)

    parameters = {
        'ddpg': {
            'parameters': {
                'input_dim': env.input_dim,
                'action_dimension': env.action_dimension,
                'gradient_clipping_norm': 5,
                'set_device': 'cpu',
                'learning_rate_actor': 0.0003,
                'learning_rate_critic': 0.002,
                'actor_noise_scale': 0.1,
                'steps_between_learning_steps': 20,
                'polyac': 0.995,
                'max_episode_length': 1000,
                'seed': seed
            },
            'total_episodes': 501
        },
        'td3': {
            'parameters': {
                'input_dim': env.input_dim,
                'action_dimension': env.action_dimension,
                'gradient_clipping_norm': 5,
                'set_device': 'cpu',
                'learning_rate_actor': 0.0003,
                'learning_rate_critic': 0.002,
                'actor_noise_scale': 0.1,
                'steps_between_learning_steps': 20,
                'polyac': 0.995,
                'max_episode_length': 1000,
                'seed': seed,
                # TD3 parameters:
                'starting_iteration_to_follow_policy': 0,
                'update_policy_and_targets_skip_rate': 2,
                'target_actor_noise_scale': 0.2,
                'target_actor_noise_clip': 0.5
            },
            'total_episodes': 501
        }
    }

    agents = {
        # 'ddpg': DDPG(**parameters['ddpg']['parameters']),
        'td3': TD3(**parameters['td3']['parameters'])
    }

    for agent_name in agents.keys():
        agent = agents[agent_name]
        # Run training
        train(
            env,
            agent,
            number_of_learning_iterations_in_one_step,
            total_episodes=parameters[agent_name]['total_episodes'],
            rolling_window_size=rolling_window_size,
            reward_curve_display_frequency=reward_curve_display_frequency,
            save_model_frequency=save_model_frequency
        )
