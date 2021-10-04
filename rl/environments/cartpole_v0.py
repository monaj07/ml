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
import gym
import numpy as np
from os.path import dirname, abspath
from PIL import Image
import sys
import torch
import torchvision.transforms as T

sys.path.append(dirname(dirname(abspath(__file__))))
from utilities import make_deterministic


class CartPoleV0:
    """
    This class instantiate the environment.
    """
    def __init__(self, seed=1364):

        # Define the environment
        self.env = gym.make('CartPole-v0').unwrapped
        # ----------------------------------------
        # Make the algorithm outputs reproducible
        make_deterministic(seed, self.env)
        # ----------------------------------------
        self.env.reset()

        # Get number of actions from gym action space
        self.num_actions = self.env.action_space.n

        # Get screen size so that we can initialize Q-network layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a cropped and down-scaled render buffer in get_screen()
        # the output of get_screen is a torch frame of shape (B, C, H, W)
        _, _, screen_height, screen_width = self.get_screen().shape
        self.input_dim = (screen_height, screen_width)

    def _get_cart_location(self, screen_width):
        # x_threshold: maximum range of the cart to each side (in terms of units)
        world_width = self.env.x_threshold * 2
        # Finding the scale of the world relative to the screen
        scale = screen_width / world_width
        # Finding the x-axis location of the center of the cart on the screen
        # by mapping its per-unit location from its current state to the screen
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]

        # Now in order to crop the screen horizontally into a smaller screen of width='view_width',
        # We check the location of the cart center;
        # If it has enough margin from both edges of the original screen width,
        # we crop a rectangle with 'view_width/2' from each side of the cart center
        # (option III in the following condition).
        # However if the cart center is closer than 'view_width/2' to the screen edges,
        # then we select one of the first two options.
        view_width = int(screen_width * 0.6)
        cart_location = self._get_cart_location(screen_width)
        if cart_location < (view_width // 2):
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)

        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([T.ToPILImage(),
                            T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])

        # Resize, and add a batch dimension (BCHW)
        final_torch_screen = resize(screen).unsqueeze(0)
        return final_torch_screen

    def run_single_episode(self, agent, explorer):
        # Make each episode deterministic based on the total_iteration_number
        make_deterministic(agent.total_steps_so_far, self.env)

        finished = False
        episode_rewards = []
        episode_losses = []

        # Create the first state of the episode
        prev_screen = self.get_screen()
        curr_screen = self.get_screen()
        state_1 = curr_screen - prev_screen

        while not finished:
            # select action (epsilon-greedy strategy)
            action_1 = explorer(self.num_actions, agent.total_steps_so_far)
            if action_1 == -1:
                # Find the greedy action
                with torch.no_grad():
                    action_1 = agent.policy_net(state_1.to(agent.device)).max(-1)[1].item()
            # Take the selected action in the environment
            _, reward_1, finished, _ = self.env.step(action_1)

            # After taking a step in the environment, the game frame is updated;
            # so we put the content of the current screen into the prev_screen,
            # and then (if game is not finished),
            # update the value of current screen with a new get_screen() call.
            prev_screen = curr_screen.clone()
            if not finished:
                # Observe the next frame and create the next state.
                # (In order to capture the dynamic of this environment,
                # we form our state by computing the difference between two frames)
                curr_screen = self.get_screen()
                state_2 = curr_screen - prev_screen
            else:
                # when episode is finished, state_2 does not matter,
                # and won't contribute to the optimisation
                # (because state_1 was the last state of the episode)
                state_2 = 0 * state_1

            # Add the current transition (s, a, r, s', done) to the replay memory
            agent.add_experience_to_replay_memory(
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
            if len(agent.replay_memory) >= agent.batch_size:
                # Take a random sample minibatch from the replay memory
                minibatch = agent.sample_from_replay_memory(agent.batch_size)
                # Compute the TD loss over the minibatch
                loss = agent.learning_step(minibatch)
                # Track the value of loss (for debugging purpose)
                episode_losses.append(loss.item())

            # Go to the next step of the episode
            state_1 = state_2
            # Add up the rewards collected during this episode
            episode_rewards.append(reward_1)
            # One single training iteration is passed
            agent.total_steps_so_far += 1

        # Return the total rewards collected within this single episode run
        return episode_rewards


class CartPoleV0Simple4D:
    """
    This class instantiate the environment with simple 4d state space.
    """
    def __init__(self, seed=1364):

        # Define the environment
        self.env = gym.make('CartPole-v0').unwrapped
        # ----------------------------------------
        # Make the algorithm outputs reproducible
        make_deterministic(seed, self.env)
        # ----------------------------------------
        self.env.reset()

        # Get number of actions from gym action space
        self.num_actions = self.env.action_space.n
        # Get the space size
        self.input_dim = self.env.state.size
        self.average_score_required_to_win = 200

    def run_single_episode(self, agent, explorer):
        # Make each episode deterministic based on the total_iteration_number
        make_deterministic(agent.total_steps_so_far, self.env)

        finished = False
        episode_rewards = []
        episode_losses = []

        # Create the first state of the episode
        state_1 = self.env.state
        state_1 = torch.from_numpy(state_1).unsqueeze(0).float()

        while not finished:
            # select action (epsilon-greedy strategy)
            action_1 = explorer(self.num_actions, agent.total_steps_so_far)
            if action_1 == -1:
                agent.policy_net.eval()
                # Find the greedy action
                with torch.no_grad():
                    action_1 = agent.policy_net(state_1.to(agent.device)).max(-1)[1].item()
                agent.policy_net.train()
            # Take the selected action in the environment
            state_2, reward_1, finished, _ = self.env.step(action_1)
            state_2 = torch.from_numpy(state_2).unsqueeze(0).float()

            if finished:
                state_2 = 0 * state_1

            # Add the current transition (s, a, r, s', done) to the replay memory
            agent.add_experience_to_replay_memory(
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
            if len(agent.replay_memory) >= agent.batch_size:
                # Take a random sample minibatch from the replay memory
                minibatch = agent.sample_from_replay_memory(agent.batch_size)
                # Compute the TD loss over the minibatch
                loss = agent.learning_step(minibatch)
                # Track the value of loss (for debugging purpose)
                episode_losses.append(loss.item())

            # Go to the next step of the episode
            state_1 = state_2
            # Add up the rewards collected during this episode
            episode_rewards.append(reward_1)
            # One single training iteration is passed
            agent.total_steps_so_far += 1

            # If the agent has received a satisfactory episode reward, stop it.
            if sum(episode_rewards) >= self.average_score_required_to_win:
                finished = True

        # Return the total rewards collected within this single episode run
        return episode_rewards


if __name__ == "__main__":
    seed = 1364
    cartpole_v0 = CartPoleV0(seed=seed)
    cartpole_v0_simple_4d = CartPoleV0Simple4D(seed=seed)
