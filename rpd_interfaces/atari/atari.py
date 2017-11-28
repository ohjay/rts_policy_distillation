#!/usr/bin/env python

"""
atari.py

Interface for Atari games.
"""

import os
import gym
from gym import wrappers
from atari_wrappers import *

from rpd_interfaces.interfaces import Environment

class AtariEnv(Environment):
    def __init__(self, env_name, expt_dir='tmp/vid_dir/'):
        self.base_env, self.env = None, None
        if env_name == 'atari_pong_ram':
            self.base_env = gym.make('Pong-ram-v0')
            self.env = wrappers.Monitor(self.base_env, os.path.join(expt_dir, 'gym'),
                                        force=True, video_callable=lambda i: False)
            self.env = wrap_deepmind_ram(self.env)
            print('input_shape: %r' % self.env.observation_space.shape)
            print('num_actions: %r' % self.env.action_space.n)
        else:
            raise ValueError('atari environment %s not supported' % env_name)

    def seed(self, i):
        self.base_env.seed(i)

    def reset(self, **kwargs):
        """Resets the environment and returns an (observation, reward, done) tuple."""
        obs = self.env.reset()
        return {'ram_in': obs}, None, None

    def get_random_action(self, **kwargs):
        return np.array((self.env.action_space.sample(),))

    def get_valid_action_from_q_values(self, q_values):
        return np.array((np.argmax(q_values),))

    def step(self, action, **kwargs):
        obs, reward, done, _ = self.env.step(action[0])
        return {'ram_in': obs}, reward, done
