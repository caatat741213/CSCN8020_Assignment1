import numpy as np

class Policy:
    """
    Base class for all policies to ensure Object-Oriented Design.
    Shows clear separation of responsibilities.
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def get_action(self, state):
        """Returns an action given the state."""
        raise NotImplementedError

    def get_action_prob(self, state, action):
        """Returns the probability of taking an action in a given state."""
        raise NotImplementedError


class RandomPolicy(Policy):
    """
    Problem 4: Behavior Policy (mu)
    A random policy used to explore the environment and generate episodes.
    """
    def __init__(self, num_actions):
        super().__init__(num_actions)

    def get_action(self, state=None):
        # Each action has equal probability (1 / |A|)
        return np.random.choice(self.num_actions)

    def get_action_prob(self, state, action):
        return 1.0 / self.num_actions


class GreedyTargetPolicy(Policy):
    """
    Problem 4: Target Policy (pi)
    A deterministic greedy policy that updates based on the max Q-value.
    """
    def __init__(self, num_states, num_actions):
        super().__init__(num_actions)
        self.policy_map = np.zeros(num_states, dtype=int)

    def update_policy(self, state, best_action):
        """Updates the greedy action for a specific state."""
        self.policy_map[state] = best_action

    def get_action(self, state):
        return self.policy_map[state]

    def get_action_prob(self, state, action):
        # Deterministic policy: 1 for the greedy action, 0 for others
        return 1.0 if self.policy_map[state] == action else 0.0