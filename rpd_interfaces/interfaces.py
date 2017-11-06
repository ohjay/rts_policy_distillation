#!/usr/bin/env python

"""
interfaces.py

Abstract class for an environment's interface.
"""

class Environment(object):
    def reset(self):
        raise NotImplementedError('blocking method for resetting (starting a new) game')

    def get_updates(self):
        raise NotImplementedError('abstract generator method for getting real-time updates')

    def extract_observation(self, update):
        raise NotImplementedError('abstract method for parsing observations from updates')

    def get_random_action(self):
        raise NotImplementedError('sample a random action from the environment')

    def apply_action(self, action):
        raise NotImplementedError('abstract method for taking action; returns (obs, reward, done) tuple')
