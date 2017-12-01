#!/usr/bin/env python

"""
dqn.py

Deep Q-network as described by CS 294-112 (goo.gl/MhA4eA).
"""

import yaml
import os, sys
import operator
import datetime
import functools, itertools
import tensorflow as tf
from collections import namedtuple

from rpd_learning.dqn_utils import *
from rpd_learning.models import Model
from rpd_learning.general_utils import rm_rf, dominant_dtype
import random


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])
_LAUNCH_TIME = datetime.datetime.now()

def learn(env, config, optimizer_spec, session, exploration=LinearSchedule(1000000, 0.1), stopping_criterion=None,
          replay_buffer_size=1000000, batch_size=32, gamma=0.99, learning_starts=50000, learning_freq=4,
          frame_history_len=4, target_update_freq=10000, grad_norm_clipping=10):
    """
    Run the DQN algorithm.
    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: rpd_interfaces.interfaces.Environment
        Environment to train on.
    config: dict
        Dictionary of useful information.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule for the optimizer.
    session: tf.Session
        TensorFlow session to use.
    exploration: rpd_learning.dqn_utils.Schedule
        Schedule for probability of choosing random action.
    stopping_criterion: (env, t) -> bool
        Should return true when it's okay for the RL algorithm to stop.
        Takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount factor.
    learning_starts: int
        After how many environment steps to start replaying experiences.
    learning_freq: int
        How many steps of environment to take between every experience replay.
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between each update to the target Q network.
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """

    # Set up checkpoint folder
    checkpoint_dir = os.path.join('.checkpoints', 'run_%s' % _LAUNCH_TIME.strftime('%m-%d__%H_%M'))
    try:
        os.makedirs(checkpoint_dir)
    except OSError:
        rm_rf(checkpoint_dir, 'ERROR: %s already exists! Delete this folder? (True/False)' % checkpoint_dir)
        os.makedirs(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, 'config_in.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)  # save the config as backup

    ###############
    # BUILD MODEL #
    ###############

    arch = config['dqn_arch']
    input_names = sorted(arch['inputs'].keys())
    assert len(input_names) > 0, 'observations are required'
    input_shapes = [arch['inputs'][input_name]['shape'] for input_name in input_names]
    input_dtypes = [arch['inputs'][input_name]['dtype'] for input_name in input_names]
    if len(set(input_dtypes)) > 1:
        print('Warning: you have inputs of different data types. You might want to keep them all the same.')
    output_names = sorted(arch['outputs'].keys(), key=lambda x: arch['outputs'][x].get('order', float('inf')))

    _obs_encoding = {}  # information to be filled in about the encoding, so we don't have to determine it every time
    _obs_encoding_dtype = dominant_dtype(input_dtypes)

    def _obs_to_np(obs):
        """Reformats observation as a single NumPy array.
        An observation, at least from the RPD interface, will be given as an {input_name: value} dict.
        """
        if 'encoding_option' in _obs_encoding:
            # Expedited route (don't need to run as many condition tests)
            encoding_option = _obs_encoding['encoding_option']
            if encoding_option == 1:
                return obs[input_names[0]]
            elif encoding_option == 2:
                return np.stack([obs[input_name] for input_name in input_names], axis=0)
            elif encoding_option == 3:
                return np.concatenate([obs[input_name] for input_name in input_names], axis=_obs_encoding['axis'])
            elif encoding_option == 4:
                return np.concatenate([obs[input_name].flatten() for input_name in input_names])

        if len(input_names) == 1:
            _obs_encoding['encoding_option'] = 1
            return obs[input_names[0]]  # if there is only one input, return as-is (1)

        num_dims0 = len(input_shapes[0])
        if all(len(_shape) == num_dims0 for _shape in input_shapes[1:]):
            # If it's possible to join the inputs along a new axis, do it (2)
            shape0 = input_shapes[0]
            if all(_shape == shape0 for _shape in input_shapes[1:]):
                _obs_encoding['encoding_option'] = 2
                return np.stack([obs[input_name] for input_name in input_names], axis=0)

            # If it's possible to join the inputs along an existing axis, do it (3)
            for dim in range(num_dims0):
                without_axis0 = shape0[:dim] + shape0[dim+1:]
                if all(_shape[:dim] + _shape[dim+1:] == without_axis0 for _shape in input_shapes[1:]):
                    _obs_encoding['encoding_option'] = 3
                    _obs_encoding['axis'] = dim
                    return np.concatenate([obs[input_name] for input_name in input_names], axis=dim)

        # Return a concatenation of flattened arrays (4)
        _obs_encoding['encoding_option'] = 4
        return np.concatenate([obs[input_name].flatten() for input_name in input_names])

    def _np_to_obs(obs_np, batched=False):
        """Separates observation into individual inputs (the {input_name: value} dict it was originally).
        Note: it is possible that OBS_NP represents not a single observation, but an entire batch of observations.

        This the inverse of `_obs_to_np`.
        """
        if 'encoding_option' in _obs_encoding:
            encoding_option = _obs_encoding['encoding_option']
            if encoding_option == 1:
                return {input_names[0]: obs_np}
            elif encoding_option == 2:
                axis = 1 if batched else 0
                individual = np.split(obs_np, obs_np.shape[axis], axis=axis)
                return {input_name: individual[i] for i, input_name in enumerate(input_names)}
            elif encoding_option == 3:
                split_indices = np.cumsum([_shape[_obs_encoding['axis']] for _shape in input_shapes])
                axis = _obs_encoding['axis'] + 1 if batched else _obs_encoding['axis']
                individual = np.split(obs_np, split_indices, axis=axis)
                return {input_name: individual[i] for i, input_name in enumerate(input_names)}
            # Encoding option #4 shall be handled below
        else:
            if len(input_names) == 1:
                _obs_encoding['encoding_option'] = 1
                return {input_names[0]: obs_np}  # inverse of (1)

            num_dims0 = len(input_shapes[0])
            if all(len(_shape) == num_dims0 for _shape in input_shapes[1:]):
                # Inverse of (2)
                shape0 = input_shapes[0]
                if all(_shape == shape0 for _shape in input_shapes[1:]):
                    _obs_encoding['encoding_option'] = 2
                    axis = 1 if batched else 0
                    individual = np.split(obs_np, obs_np.shape[axis], axis=axis)
                    return {input_name: individual[i] for i, input_name in enumerate(input_names)}

                # Inverse of (3)
                for dim in range(num_dims0):
                    without_axis0 = shape0[:dim] + shape0[dim + 1:]
                    if all(_shape[:dim] + _shape[dim + 1:] == without_axis0 for _shape in input_shapes[1:]):
                        _obs_encoding['encoding_option'] = 3
                        _obs_encoding['axis'] = dim
                        split_indices = np.cumsum([_shape[dim] for _shape in input_shapes])
                        axis = dim + 1 if batched else dim
                        individual = np.split(obs_np, split_indices, axis=axis)
                        return {input_name: individual[i] for i, input_name in enumerate(input_names)}

        # Inverse of (4)
        _obs_encoding['encoding_option'] = 4
        obs, i = {}, 0
        for input_name in input_names:
            _shape = arch['inputs'][input_name]['shape']
            size = functools.reduce(operator.mul, _shape, 1)
            if batched:
                _shape = np.insert(_shape, 0, obs_np.shape[0])
                try:
                    obs[input_name] = np.reshape(obs_np[:, i:i+size], _shape)
                except ValueError as e:
                    print('i: %d' % i)
                    print('obs_np shape: %r' % (obs_np.shape,))
                    print('shape, size: %r, %d' % (_shape, size))
                    raise
            else:
                obs[input_name] = np.reshape(obs_np[i:i+size], _shape)
            i += size
        return obs

    # Set up placeholders
    obs_t_ph = {input_name: None for input_name in input_names}  # current observation (or state)
    obs_t_ph_float = {input_name: None for input_name in input_names}
    obs_tp1_ph = {input_name: None for input_name in input_names}  # next observation (or state)
    obs_tp1_ph_float = {input_name: None for input_name in input_names}
    for input_name, dtype, shape in zip(input_names, input_dtypes, input_shapes):
        obs_t_ph[input_name] = tf.placeholder(getattr(tf, dtype), [None] + list(shape))
        obs_tp1_ph[input_name] = tf.placeholder(getattr(tf, dtype), [None] + list(shape))
        if dtype == 'float32':
            obs_t_ph_float[input_name] = obs_t_ph[input_name]
            obs_tp1_ph_float[input_name] = obs_tp1_ph[input_name]
        else:  # casting to float on GPU ensures lower data transfer times
            obs_t_ph_float[input_name] = tf.cast(obs_t_ph[input_name], tf.float32) / 255.0
            obs_tp1_ph_float[input_name] = tf.cast(obs_tp1_ph[input_name], tf.float32) / 255.0
    act_t_ph = {output_name: tf.placeholder(tf.int32, [None]) for output_name in output_names}  # current action
    rew_t_ph = tf.placeholder(tf.float32, [None])  # current reward
    done_mask_ph = tf.placeholder(tf.float32, [None])  # end of episode mask (1 if next state = end of an episode)

    # Create networks (for current/next Q-values)
    q_func = Model(arch, inputs=obs_t_ph_float, scope='q_func', reuse=False)
    target_q_func = Model(arch, inputs=obs_tp1_ph_float, scope='target_q_func', reuse=False)

    # Compute the Bellman error
    all_q_j = []
    all_y_j = []
    for output_name in output_names:
        _num = arch['outputs'][output_name]['shape'][0]
        q_out = q_func.outputs[output_name]
        target_out = target_q_func.outputs[output_name]
        all_q_j.append(tf.reduce_sum(tf.multiply(tf.one_hot(act_t_ph[output_name], _num), q_out), axis=1))
        all_y_j.append(rew_t_ph + tf.multiply(gamma, tf.reduce_max(tf.stop_gradient(target_out), axis=1)))

    total_error = 0
    for q_j, y_j in zip(all_q_j, all_y_j):
        # scalar valued tensor representing Bellman error (error based on current and next Q-values)
        total_error += tf.reduce_mean(tf.square(y_j - q_j))
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # Construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error, var_list=q_func_vars, clip_val=grad_norm_clipping)

    # `update_target_fn` will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###########
    # RUN ENV #
    ###########

    train_params = config.get('train_params', {})
    model_initialized = False
    moments_initialized = False
    obs_mean, obs_std = {}, {}
    num_param_updates = 0
    reset_kwargs = config.get('reset_kwargs', {})
    step_kwargs = config.get('step_kwargs', {})
    get_action_kwargs = config.get('get_action_kwargs', {})
    get_random_kwargs = config.get('get_random_kwargs', {})
    last_obs, reward, done = env.reset(**reset_kwargs)
    last_obs_np = _obs_to_np(last_obs)
    normalize_inputs = train_params.get('normalize_inputs', True)
    log_freq = train_params.get('log_freq', 150)
    play_count = 0
    game_steps = 0

    last_episode_rewards = []
    episode_rewards = []
    episode_returns = []
    mean_episode_return = -float('nan')
    best_mean_episode_return = float('-inf')

    save_images = False

    with open(os.path.join(checkpoint_dir, 'log.txt'), 'a+') as logfile:
        def print_and_log(text):
            """Print TEXT to standard output, while also writing it to the log file."""
            print(text)
            logfile.write(text + '\n')

        for t in itertools.count():
            if stopping_criterion is not None and stopping_criterion(env, t):
                break

            # Step the env and store the transition in the replay buffer
            idx = replay_buffer.store_frame(last_obs_np, dtype=_obs_encoding_dtype)

            # Choose action via epsilon greedy exploration
            eps = exploration.value(t)
            if not model_initialized or random.random() < eps:
                # If first step, choose a random action
                action = env.get_random_action(**get_random_kwargs)
            else:
                obs_recent = _np_to_obs(replay_buffer.encode_recent_observation())
                feed_dict = {}
                for input_name in obs_recent.keys():
                    if normalize_inputs:
                        obs_recent[input_name] = (obs_recent[input_name] - obs_mean[input_name]) / obs_std[input_name]
                    shaped_val = np.reshape(obs_recent[input_name], (1,) + obs_recent[input_name].shape)
                    feed_dict[obs_t_ph[input_name]] = shaped_val

                fetches = [q_func.outputs[output_name] for output_name in output_names]
                q_values = session.run(fetches, feed_dict=feed_dict)
                action = env.get_action_from_q_values(q_values, **get_action_kwargs)

            last_obs, reward, done = env.step(action, **step_kwargs)
            replay_buffer.store_effect(idx, action, reward, done)
            if save_images and len(last_obs):
                image = env.get_image_of_state(last_obs)
                if image is not None:
                    run_dir = os.path.join('img', 'run_%s' % _LAUNCH_TIME.strftime('%m-%d__%H_%M'))
                    if not os.path.exists(run_dir):
                        os.makedirs(run_dir)
                        print('Saving images to %s.\n' % run_dir)
                    image.save("{}/Game_{}_Step_{}.png".format(run_dir, play_count, game_steps))

            if done:
                last_obs, reward, done = env.reset(**reset_kwargs)
                last_obs_np = _obs_to_np(last_obs)

                last_episode_rewards = episode_rewards
                episode_returns.append(sum(episode_rewards))
                episode_rewards = []
                save_images = False
                play_count += 1
                game_steps = 0
            else:
                last_obs_np = _obs_to_np(last_obs)
                episode_rewards.append(reward)
                game_steps += 1

            # At this point, the environment should have been advanced one step (and reset if `done` was true),
            # `last_obs_np` should point to the new latest observation,
            # and the replay buffer should contain one more transition.

            # Perform experience replay and train the network
            # (once the replay buffer contains enough samples for us to learn something useful)
            if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size):
                # Use replay buffer to sample a batch of transitions
                obs_batch_np, act_batch, rew_batch, next_obs_batch_np, done_mask = replay_buffer.sample(batch_size)
                obs_batch = _np_to_obs(obs_batch_np, batched=True)
                next_obs_batch = _np_to_obs(next_obs_batch_np, batched=True)

                if normalize_inputs:
                    if not moments_initialized:
                        # Compute observation mean and standard deviation (for use in normalization)
                        _obs_np, _, _, _, _ = replay_buffer.sample(replay_buffer.num_in_buffer - 1)
                        _obs = _np_to_obs(_obs_np, batched=True)
                        obs_mean = {input_name: np.mean(_obs[input_name], axis=0) for input_name in _obs.keys()}
                        obs_std = {input_name: np.std(_obs[input_name], axis=0) + 1e-9 for input_name in _obs.keys()}
                        _obs_np, _obs = None, None
                        moments_initialized = True
                    obs_t_feed = {obs_t_ph[_name]: (obs_batch[_name] - obs_mean[_name]) / obs_std[_name]
                                  for _name in obs_batch.keys()}
                    obs_tp1_feed = {obs_tp1_ph[_name]: (next_obs_batch[_name] - obs_mean[_name]) / obs_std[_name]
                                    for _name in next_obs_batch.keys()}
                else:
                    obs_t_feed = {obs_t_ph[_name]: obs_batch[_name] for _name in obs_batch.keys()}
                    obs_tp1_feed = {obs_tp1_ph[_name]: next_obs_batch[_name] for _name in next_obs_batch.keys()}

                # Initialize the model
                if not model_initialized:
                    initialize_interdependent_variables(session, tf.global_variables(),
                                                        merge_dicts(obs_t_feed, obs_tp1_feed))
                    model_initialized = True

                # Train the model
                session.run(train_fn, merge_dicts(
                    obs_t_feed,
                    obs_tp1_feed,
                    {act_t_ph[output_name]: act_batch[:, i] for i, output_name in enumerate(output_names)},
                    {rew_t_ph: rew_batch, done_mask_ph: done_mask, learning_rate: optimizer_spec.lr_schedule.value(t)}
                ))

                # Periodically update the target network
                if num_param_updates % target_update_freq == 0:
                    session.run(update_target_fn)
                num_param_updates += 1

            # Log progress
            if len(episode_returns) > 0:
                mean_episode_return = np.mean(episode_returns[-100:])
            if len(episode_returns) > 100:
                best_mean_episode_return = max(best_mean_episode_return, mean_episode_return)
            if play_count % log_freq == 0 and game_steps == 0 and model_initialized:
                mean_return_output = 'mean return (100 episodes) %f' % mean_episode_return
                print_and_log(('step %d ' % t).ljust(len(mean_return_output), '-'))
                print_and_log(mean_return_output)
                print_and_log('best mean return (100 episodes) %f' % best_mean_episode_return)
                print_and_log('episodes %d' % len(episode_returns))
                print_and_log('exploration %f' % exploration.value(t))
                print_and_log('learning_rate %f' % optimizer_spec.lr_schedule.value(t))

                if train_params.get('log_recent_rewards', True):
                    print_and_log('mean of recent rewards %.2f' % np.mean(last_episode_rewards))
                    print_and_log('recent rewards %r' % last_episode_rewards)

                # Save network parameters and images from next episode
                if train_params.get('save_model', True):
                    q_func.save(session, t, outfolder=os.path.join(checkpoint_dir, 'q_func'))
                    target_q_func.save(session, t, outfolder=os.path.join(checkpoint_dir, 'target'))
                save_images = train_params.get('save_images', True)

                print_and_log('')  # newline
                sys.stdout.flush()
                logfile.flush()
