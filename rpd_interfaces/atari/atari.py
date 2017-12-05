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

_PONG_INDEX = 3

class AtariEnv(Environment):
    def __init__(self, env_name, expt_dir='tmp/vid_dir/', monitor=True):
        self.base_env, self.env = None, None
        self.obs_input_name = None
        if env_name == 'atari_pong':
            benchmark = gym.benchmark_spec('Atari40M')
            task = benchmark.tasks[_PONG_INDEX]
            print('max timesteps: %d' % task.max_timesteps)
            self.base_env = gym.make(task.env_id)
            self.env = self.base_env
            if monitor:
                self.env = wrappers.Monitor(self.env, os.path.join(expt_dir, 'gym'),
                                            force=True, video_callable=lambda i: False)
            self.env = wrap_deepmind(self.env)
            img_h, img_w, img_c = self.env.observation_space.shape
            print('img_h: %d' % img_h)
            print('img_w: %d' % img_w)
            print('img_c: %d' % img_c)
            print('num_actions: %r' % self.env.action_space.n)
            self.obs_input_name = 'img_in'
        elif env_name == 'atari_pong_ram':
            self.base_env = gym.make('Pong-ram-v0')
            self.env = self.base_env
            if monitor:
                self.env = wrappers.Monitor(self.env, os.path.join(expt_dir, 'gym'),
                                            force=True, video_callable=lambda i: False)
            self.env = wrap_deepmind_ram(self.env)
            print('input_shape: %r' % self.env.observation_space.shape)
            print('num_actions: %r' % self.env.action_space.n)
            self.obs_input_name = 'ram_in'
        else:
            raise ValueError('atari environment %s not supported' % env_name)

    def seed(self, i):
        self.base_env.seed(i)

    def reset(self, **kwargs):
        """Resets the environment and returns an (observation, reward, done) tuple."""
        obs = self.env.reset()
        return {self.obs_input_name: obs}, None, None

    def get_random_action(self, **kwargs):
        return np.array((self.env.action_space.sample(),))

    def get_action_from_q_values(self, q_values, **kwargs):
        return np.array((np.argmax(q_values),))

    def step(self, action, **kwargs):
        obs, reward, done, _ = self.env.step(action[0])
        return {self.obs_input_name: obs}, reward, done
