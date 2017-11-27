#!/usr/bin/env python

import os
import yaml
import argparse
import tensorflow as tf

from rpd_learning.dqn_utils import PiecewiseSchedule
from rpd_learning import dqn

SUPPORTED_ENVS = {'generals', 'generals_sim',}
SUPPORTED_OPS = {'train',}

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
    session = get_session()
    num_timesteps = int(4e7)
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 10.0
    lr_schedule = PiecewiseSchedule([
        (0,                   1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2,  5e-5 * lr_multiplier),
    ], outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer, kwargs=dict(epsilon=1e-4), lr_schedule=lr_schedule)

    def stopping_criterion(env, t):
        return t > 8e7  # TODO idk

    exploration_schedule = PiecewiseSchedule([(0, 0.8), (1e6, 0.2), (num_iterations / 2, 0.1), (num_iterations, 0.01)], outside_value=0.01)

    dqn.learn(env, config, optimizer_spec=optimizer, session=session, exploration=exploration_schedule,
              stopping_criterion=stopping_criterion, replay_buffer_size=1000000, batch_size=32, gamma=0.99,
              learning_starts=50000, learning_freq=1, frame_history_len=1, target_update_freq=10000, grad_norm_clipping=10)

    if hasattr(env, 'close'):  # cleanup, if applicable
        env.close()

def test(env, config):
    for update in env.get_updates():
        observation = env.extract_observation(update)

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
        from rpd_interfaces.generals.simulator import GeneralsEnv
        env = GeneralsEnv('replays_prod/')


    eval(config['operation'])(env, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    run(config)
