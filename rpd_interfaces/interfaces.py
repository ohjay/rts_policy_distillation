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

    def reset(self):
        raise NotImplementedError('blocking method for resetting (starting a new) game')

    def get_random_action(self):
        raise NotImplementedError('sample a random action from the environment')

    def apply_action(self, action):
        raise NotImplementedError('abstract method for taking action; returns (obs, reward, done) tuple')
