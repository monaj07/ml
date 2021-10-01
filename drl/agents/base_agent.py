"""
Implementing an abstract class for all agents.
"""
import gym
import logging
import numpy as np
import os
import random
import sys
import time
import torch


class Base_Agent(object):
    def __init__(self, config):
        self.logger = self.setup_logger()
        self.config = config
        self.set_random_seeds(config.seed)
        self.environment = config.environment
        self.environment_title = self.get_environment_title()
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = int(self.get_action_size())
        self.config.action_size = self.action_size

        self.state_size = int(self.get_state_size())
        self.hyperparameters = config.hyperparameters
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.rolling_score_window = self.get_trials()
        # self.max_steps_per_episode = self.environment.spec.max_episode_steps
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning
        self.log_game_info()

    def run_one_episode(self):
        """Run through one episode in the game.
        This method must be overridden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def get_environment_title(self):
        """Extracts name of environment from it"""
        try:
            name = self.environment.unwrapped.id
        except AttributeError:
            try:
                if str(self.environment.unwrapped)[1:11] == "FetchReach":
                    return "FetchReach"
                elif str(self.environment.unwrapped)[1:8] == "AntMaze":
                    return "AntMaze"
                elif str(self.environment.unwrapped)[1:7] == "Hopper":
                    return "Hopper"
                elif str(self.environment.unwrapped)[1:9] == "Walker2d":
                    return "Walker2d"
                else:
                    name = self.environment.spec.id.split("-")[0]
            except AttributeError:
                name = str(self.environment.env)
                if name[0:10] == "TimeLimit<":
                    name = name[10:]
                name = name.split(" ")[0]
                if name[0] == "<":
                    name = name[1:]
                if name[-3:] == "Env":
                    name = name[:-3]
        return name

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        if "overwrite_action_size" in self.config.__dict__:
            return self.config.overwrite_action_size
        if "action_size" in self.environment.__dict__:
            return self.environment.action_size
        if self.action_types == "DISCRETE":
            return self.environment.action_space.n
        else:
            return self.environment.action_space.shape[0]

    def get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
        random_state = self.environment.reset()
        if isinstance(random_state, dict):
            state_size = random_state["observation"].shape[0] + random_state["desired_goal"].shape[0]
            return state_size
        else:
            return random_state.size

    def get_score_required_to_win(self):
        """Gets average score required to win game"""
        print("TITLE ", self.environment_title)
        if self.environment_title == "FetchReach":
            return -5
        if self.environment_title in ["AntMaze", "Hopper", "Walker2d"]:
            print("Score required to win set to infinity therefore no learning rate annealing will happen")
            return float("inf")
        try:
            return self.environment.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.environment.spec.reward_threshold
            except AttributeError:
                return self.environment.unwrapped.spec.reward_threshold

    def get_trials(self):
        """Gets the number of trials to average a score over"""
        if self.environment_title in ["AntMaze", "FetchReach", "Hopper", "Walker2d", "CartPole"]:
            return 100
        try:
            return self.environment.unwrapped.trials
        except AttributeError:
            return self.environment.spec.trials

    def setup_logger(self):
        """Sets up the logger"""
        filename = "Training.log"
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except:
            pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    def log_game_info(self):
        """Logs info relating to the game"""
        for ix, param in enumerate(
                [self.environment_title,
                 self.action_types,
                 self.action_size,
                 self.state_size,
                 self.hyperparameters,
                 self.average_score_required_to_win,
                 self.rolling_score_window,
                 self.device]
        ):
            self.logger.info("{} -- {}".format(ix, param))

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config.seed)
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys():
            self.exploration_strategy.reset()
        self.logger.info("Reseting game -- New start state {}".format(self.state))

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.run_one_episode()
            self.calculate_and_print_rewards()
        time_taken = time.time() - start
        if show_whether_achieved_goal:
            self.show_whether_achieved_goal()
        if self.config.save_model:
            self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def take_action(self, action):
        """take an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)

    def calculate_and_print_rewards(self):
        # Calculate the results of this episode (rolling & max reward)
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(
            np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:])
        )

        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

        text = f"""\r Episode {len(self.game_full_episode_scores)}, """
        text += f""" Score: {self.game_full_episode_scores[-1]:.2f}, """
        text += f""" Max score seen: {self.max_episode_score_seen:.2f}, """
        text += f""" Rolling score: {self.rolling_results[-1]:.2f}, """
        text += f""" Max rolling score seen: {self.max_rolling_score_seen:.2f}"""
        sys.stdout.write(text)
        sys.stdout.flush()

    def show_whether_achieved_goal(self):
        """Prints out whether the agent achieved the environment target goal"""
        index_achieved_goal = self.achieved_required_score_at_index()
        print(" ")
        if index_achieved_goal == -1: #this means agent never achieved goal
            print("\033[91m" + "\033[1m" +
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" +
                  "{} achieved required score at episode {} \n".format(self.agent_name, index_achieved_goal) +
                  "\033[0m" + "\033[0m")

    def achieved_required_score_at_index(self):
        """Returns the episode at which agent achieved goal or -1 if it never achieved it"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required_to_win:
                return ix
        return -1

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        if random.random() < 0.001:
            self.logger.info("Learning rate {}".format(new_lr))

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        self.logger.info("Loss -- {}".format(loss.item()))
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def create_neural_net(self, input_dim, output_dim):
        """Creates a neural network for the agents to use"""
        raise NotImplementedError("The derived agent must implement its network")
