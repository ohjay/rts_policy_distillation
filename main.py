#!/usr/bin/env python

import os
import copy
import yaml
import argparse
import random
import numpy as np
import tensorflow as tf

from rpd_learning.dqn_utils import PiecewiseSchedule
from rpd_learning import dqn

SUPPORTED_ENVS = {'generals', 'generals_sim', 'atari_pong', 'atari_pong_ram'}
SUPPORTED_OPS = {'train'}

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUs: ", get_available_gpus())
    return session

def train(env, config):
    # Run DQN training for the given environment
    train_params = config.get('train_params', {})

    if 'random_seed' in train_params:
        random.seed(train_params['random_seed'])
        np.random.seed(train_params['random_seed'])
        tf.set_random_seed(train_params['random_seed'])
        env.seed(train_params['random_seed'])

    session = get_session()
    num_timesteps = train_params.get('num_timesteps', int(4e7))
    num_iterations = float(num_timesteps) / 4.0

    def _process_endpoints(endpoints, _locals=None):
        """Convert ENDPOINTS into the format required for a PiecewiseSchedule.
        Turn a list of single-element {t: value} dictionaries (sorted by t) into a list of (t, value) tuples.
        """
        if _locals is None:
            _locals = globals()
        processed_endpoints = []
        for e in endpoints:
            t, value = e.keys()[0], e.values()[0]
            processed_endpoints.append((
                eval(t, globals(), _locals) if type(t) == str else t,
                eval(value, globals(), _locals) if type(value) == str else value
            ))
        return processed_endpoints

    lr_multiplier = train_params.get('lr_multiplier', 1.0)
    lr_endpoints = train_params.get('lr_schedule', [
        {0:                   1e-4 * lr_multiplier},
        {num_iterations / 10: 1e-4 * lr_multiplier},
        {num_iterations / 2:  5e-5 * lr_multiplier},
    ])
    print('Loaded learning rate schedule as %r.' % lr_endpoints)
    lr_endpoints = _process_endpoints(lr_endpoints, locals())
    lr_schedule = PiecewiseSchedule(lr_endpoints, outside_value=lr_endpoints[-1][1])
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer, kwargs=dict(epsilon=1e-4), lr_schedule=lr_schedule)

    def stopping_criterion(env, t):
        return t > num_timesteps * 2

    exploration_endpoints = train_params.get('exploration_schedule', [
        {0:                  1.00},
        {1e6:                0.10},
        {num_iterations / 2: 0.01}
    ])
    print('Loaded exploration schedule as %r.' % exploration_endpoints)
    exploration_endpoints = _process_endpoints(exploration_endpoints, locals())
    exploration_schedule = PiecewiseSchedule(exploration_endpoints, outside_value=exploration_endpoints[-1][1])

    learn_params = {p: train_params[p] for p in {
        'replay_buffer_size', 'batch_size', 'gamma', 'learning_starts', 'learning_freq', 'frame_history_len',
        'target_update_freq', 'grad_norm_clipping'
    } if p in train_params}
    print('Loaded train parameters as %r.' % learn_params)

    dqn.learn(env, config, optimizer_spec=optimizer, session=session, exploration=exploration_schedule,
              stopping_criterion=stopping_criterion, **learn_params)

    if hasattr(env, 'close'):  # cleanup, if applicable
        env.close()

def test(env, config):
    pass  # TODO add evaluation code

def run(config):
    """Run RPD code according to the passed-in config file."""
    assert config['env'] in SUPPORTED_ENVS, 'not a supported environment'
    assert config['operation'] in SUPPORTED_OPS, 'not a supported op'

    env = None
    env_info = config.get(config['env'], None)
    if config['env'] == 'generals':
        from rpd_interfaces.generals import generals
        user_id = env_info['user_id']
        env = generals.Generals(user_id)
    elif config['env'] == 'generals_sim':
        from rpd_interfaces.generals import generals_sim
        reward_fn_name = env_info.get('reward_fn_name', None)
        action_space = copy.deepcopy(config['dqn_arch']['outputs'])
        env = generals_sim.GeneralsEnv('replays_prod/', reward_fn_name=reward_fn_name, action_space=action_space)
    elif config['env'].startswith('atari_pong'):
        from rpd_interfaces.atari import atari
        env = atari.AtariEnv(config['env'])

    eval(config['operation'])(env, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    run(config)
