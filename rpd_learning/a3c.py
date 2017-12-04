#!/usr/bin/env python

"""
a3c.py

The asynchronous advantage actor-critic (A3C) algorithm.
Adapted from Ray's RLLib implementation (https://github.com/ray-project/ray/tree/master/python/ray/rllib),
which is in turn based on the OpenAI starter agent (https://github.com/openai/universe-starter-agent).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ray
import pickle
import random
import numpy as np
import scipy.signal
from collections import namedtuple, deque

from rpd_learning.obs_codecs import StandardCodec
from rpd_learning.models import A3CPolicy


def _print_metrics_from_workers(workers):
    all_metrics = [w.get_completed_rollout_metrics.remote() for w in workers]
    print('-----------')
    for metrics in all_metrics:
        _metrics = ray.get(metrics)
        print('---\nmean return (100 episodes) %f\nbest mean return (100 episodes) %f\n'
              % (_metrics['mean_episode_return'], _metrics['best_mean_episode_return']))

def _save(logdir, iteration, parameters):
    checkpoint_path = os.path.join(logdir, 'checkpoint-{}'.format(iteration))
    pickle.dump(parameters, open(checkpoint_path, 'wb'))
    return checkpoint_path

def _restore(checkpoint_path, policy):
    parameters = pickle.load(open(checkpoint_path, 'rb'))
    policy.set_weights(parameters)


def learn(num_workers, max_batches, env, config, batch_size=32):
    """Run the A3C algorithm."""

    ##################
    # PROCESS CONFIG #
    ##################

    arch = config['dqn_arch']
    train_params = config.get('train_params', {})
    log_freq = train_params.get('log_freq', 150)

    #########
    # TRAIN #
    #########

    ray.init()
    policy = A3CPolicy(arch)
    # remote_env = ray.put(env)
    remote_config = ray.put(config)
    workers = [Runner.remote(remote_config) for _ in range(num_workers)]
    parameters = policy.get_weights()
    remote_params = ray.put(parameters)
    ray.get([worker.set_weights.remote(remote_params) for worker in workers])

    # Start gradient calculation tasks on each worker
    gradient_list = {worker.compute_gradient.remote(): worker for worker in workers}
    batches_so_far = len(gradient_list)

    while gradient_list:
        [done_id], _ = ray.wait(list(gradient_list))
        gradient, info = ray.get(done_id)
        worker = gradient_list.pop(done_id)
        policy.apply_gradients(gradient)
        parameters = policy.get_weights()

        if batches_so_far < max_batches:
            batches_so_far += 1
            worker.set_weights.remote(parameters)
            gradient_list[worker.compute_gradient.remote()] = worker

        if batches_so_far % log_freq == 0:
            print('batch %d' % batches_so_far)
            _print_metrics_from_workers(workers)

    print('Training complete.')


class Runner(object):
    """Actor object to start running simulation on workers.
    The gradient computation is also executed from this object.
    Attributes:
        policy: Copy of graph used for policy. Used by sampler and gradients.
        rew_filter: Reward filter used in rollout post-processing.
        sampler: Component for interacting with environment and generating
            rollouts.
    """
    def __init__(self, config):
        from rpd_interfaces.atari import atari
        self.env = atari.AtariEnv(config['env'], monitor=config.get(config['env'], {}).get('monitor', True))
        # self.env = env
        self.config = config
        self.policy = A3CPolicy(self.config['dqn_arch'])

        arch = config['dqn_arch']
        input_names = sorted(arch['inputs'].keys())
        assert len(input_names) > 0, 'observations are required'
        input_shapes = [arch['inputs'][input_name]['shape'] for input_name in input_names]
        input_dtypes = [arch['inputs'][input_name]['dtype'] for input_name in input_names]
        if len(set(input_dtypes)) > 1:
            print('Warning: you have inputs of different data types. You might want to keep them all the same.')
        self._codec = StandardCodec(input_names, input_shapes, input_dtypes)

        self.reset_kwargs = config.get('reset_kwargs', {})
        self.step_kwargs = config.get('step_kwargs', {})
        self.get_random_kwargs = config.get('get_random_kwargs', {})

        self.mean_episode_return = -float('nan')
        self.best_mean_episode_return = float('-inf')
        self.episode_returns = []

    # TODO: make this asynchronous
    def get_data(self, max_timesteps=1000, eps=0.1):
        """Grab a rollout that is at most MAX_TIMESTEPS timesteps long."""
        rollout = {'state': [], 'action': [], 'reward': [], 'value': [], 'is_terminal': False, 'last_r': 0}
        episode_return = 0

        last_obs, reward, done = self.env.reset(**self.reset_kwargs)
        last_obs = last_obs.values()[0]  # TODO hacky
        last_obs_prev4 = deque(maxlen=4)
        last_obs_prev4.append(last_obs)
        for t in range(max_timesteps):
            # Choose an action
            if random.random() < eps or len(last_obs_prev4) < 4:
                action = self.env.get_random_action(**self.get_random_kwargs)[0]  # random action
                # pi_info = {'value': self.policy.value(last_obs)}
                if len(last_obs_prev4) == 4:
                    pi_info = {'value': self.policy.value(np.concatenate(last_obs_prev4, axis=2))}
            else:
                # action, pi_info = self.policy.compute_action(last_obs)
                action, pi_info = self.policy.compute_action(np.concatenate(last_obs_prev4, axis=2))
            obs, reward, done = self.env.step(np.array([action]), **self.step_kwargs)  # TODO hacky action processing
            obs = obs.values()[0]  # TODO hacky

            if len(last_obs_prev4) == 4:
                rollout['state'].append(np.concatenate(last_obs_prev4, axis=2))
                rollout['action'].append(action)
                rollout['reward'].append(reward)
                rollout['value'].append(pi_info['value'])
            episode_return += reward
            if done:
                rollout['is_terminal'] = True
                break
            last_obs = obs
            last_obs_prev4.append(last_obs)

        if not rollout['is_terminal'] and len(last_obs_prev4) == 4:
            # rollout['last_r'] = self.policy.value(last_obs)
            rollout['last_r'] = self.policy.value(np.concatenate(last_obs_prev4, axis=2))

        self.episode_returns.append(episode_return)
        if len(self.episode_returns) > 0:
            self.mean_episode_return = np.mean(self.episode_returns[-100:])
        if len(self.episode_returns) > 100:
            self.best_mean_episode_return = max(self.best_mean_episode_return, self.mean_episode_return)

        return rollout

    def get_completed_rollout_metrics(self):
        """Returns metrics on previously completed rollouts.
        """
        return {
            'mean_episode_return': self.mean_episode_return,
            'best_mean_episode_return': self.best_mean_episode_return,
        }

    def compute_gradient(self):
        rollout = {'state': []}
        while len(rollout['state']) == 0:
            rollout = self.get_data()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
        gradient, info = self.policy.compute_gradients(batch)
        return gradient, info

    def set_weights(self, params):
        self.policy.set_weights(params)

Runner = ray.remote(Runner)

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """Given a rollout, compute its returns and the advantage."""
    batch_si = np.asarray(rollout['state'])
    batch_a = np.asarray(rollout['action'])
    rewards = np.asarray(rollout['reward'])
    vpred_t = np.asarray(rollout['value'] + [rollout['last_r']])

    rewards_plus_v = np.asarray(rollout['reward'] + [rollout['last_r']])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)  # generalized advantage estimation, https://arxiv.org/abs/1506.02438

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout['is_terminal'])

Batch = namedtuple(
    'Batch', ['si', 'a', 'adv', 'r', 'terminal'])
