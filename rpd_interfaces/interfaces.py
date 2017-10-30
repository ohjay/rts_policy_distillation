#!/usr/bin/env python

"""
interfaces.py

Abstract class for an environment's interface.
"""

class Interface(object):
    def get_updates(self):
        raise NotImplementedError('abstract generator method for getting real-time updates')

    def extract_observation(self, update):
        raise NotImplementedError('abstract method for parsing observations from updates')

    # TODO - something for making actions
