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
import math
from os.path import dirname, abspath
import random
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
from utilities import make_deterministic


class ActionExplorer:
    """
    This class implements the epsilon greedy strategy and
    tells the agent when to explore and when to exploit
    (which is when the returned action of this class is -1)
    """
    def __init__(self, epsilon_decay=0.005, seed=1364):

        # Epsilon parameters
        self.current_epsilon = 0.99
        self.epsilon_start = 0.99
        self.epsilon_end = 0.05
        self.epsilon_decay = epsilon_decay

        # ----------------------------------------
        # Make the algorithm outputs reproducible
        make_deterministic(seed)
        # ----------------------------------------

    def __call__(self, num_actions, total_steps_so_far):
        # Epsilon-greedy strategy
        self.current_epsilon = self.epsilon_end + (
                (self.epsilon_start - self.epsilon_end) *
                math.exp(-self.epsilon_decay * total_steps_so_far)
        )
        z = random.uniform(0, 1)
        if z < self.current_epsilon:
            # Take a random action
            return random.choice(range(num_actions))
        else:
            # Let the output know to take the policy action
            return -1


if __name__ == "__main__":
    seed = 1364
    explorer = ActionExplorer(seed=seed)
