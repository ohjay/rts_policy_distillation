#!/usr/bin/env python

"""
interfaces.py

Abstract class for an environment's interface.
"""

class Environment(object):
    """
    An environment is an abstract entity to which actions are applied,
    and from which states/observations and rewards are given.
    """

    def seed(self, i):
        """Set random seed for the environment. If not supported, does nothing."""
        pass

    def reset(self, **kwargs):
        raise NotImplementedError('blocking method for resetting / starting a new game; '
                                  'returns (obs, reward, done) tuple')

    def get_random_action(self, **kwargs):
        raise NotImplementedError('sample a random action from the environment')

    def get_action_from_q_values(self, q_values, **kwargs):
        raise NotImplementedError('get an action from a collection of Q values')

    def step(self, action, **kwargs):
        raise NotImplementedError('abstract method for taking action; returns (obs, reward, done) tuple')

    def get_image_of_state(self, state):
        """Gets an image of the state. If not supported, return `None`."""
        return None
