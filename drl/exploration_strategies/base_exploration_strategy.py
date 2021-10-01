class BaseExplorationStrategy(object):
    """Base abstract class for agent exploration strategies. Every exploration strategy must inherit from this class
    and implement the methods perturb_action_for_exploration_purposes"""
    def __init__(self, config):
        self.config = config

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        raise ValueError("Must be implemented")