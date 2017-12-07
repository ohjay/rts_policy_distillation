#!/usr/bin/env python

"""
dqn.py

Deep Q-network as described by CS 294-112 (goo.gl/MhA4eA).
"""

import copy
import yaml
import os, sys
import datetime
import itertools
import tensorflow as tf
from collections import namedtuple, deque

from rpd_learning.dqn_utils import *
from rpd_learning.models import Model
from rpd_learning.general_utils import rm_rf, eval_keys
from rpd_learning.obs_codecs import StandardCodec
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
    print('Logging training information to `%s`.' % checkpoint_dir)

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
    _codec = StandardCodec(input_names, input_shapes, input_dtypes)

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
    total_error = 0  # scalar-valued tensor representing Bellman error (error based on current and next Q-values)
    for output_name in output_names:
        _num = arch['outputs'][output_name]['shape'][0]
        q_out = q_func.outputs[output_name]
        target_out = target_q_func.outputs[output_name]
        q_j = tf.reduce_sum(tf.multiply(tf.one_hot(act_t_ph[output_name], _num), q_out), axis=1)
        y_j = rew_t_ph + tf.multiply(gamma, tf.reduce_max(tf.stop_gradient(target_out), axis=1))
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
    reward_curriculum = eval_keys(config.get('reward_curriculum', {}))
    if reward_curriculum:
        print('[o] Loaded reward curriculum as %r.' % reward_curriculum)
    else:
        print('[o] No reward curriculum found.')
    last_obs, reward, done = env.reset(**reset_kwargs)
    last_obs_np = _codec.obs_to_np(last_obs)
    normalize_inputs = train_params.get('normalize_inputs', True)
    log_freq = train_params.get('log_freq', 150)
    save_images = train_params.get('save_images', True)
    evaluate_network = train_params.get('evaluate_network', False)
    play_count = 0
    game_steps = 0

    last_episode_rewards = []
    episode_rewards = []
    episode_returns = deque(maxlen=100)
    mean_episode_return = -float('nan')
    best_mean_episode_return = float('-inf')

    nw_episode_returns = deque(maxlen=100)  # episode returns for pure network evaluation
    nw_best_mean_episode_return, nw_best_iteration = float('-inf'), 0

    if train_params.get('restore', False):
        _restore_meta = train_params.get('restore_from', {})
        _r_run_dir, _r_iteration = _restore_meta.get('run_dir', None), _restore_meta.get('iteration', None)
        if None not in (_r_run_dir, _r_iteration):
            _r_run_dir = os.path.join('.checkpoints', _r_run_dir)
            q_func.restore(session, _r_iteration, os.path.join(_r_run_dir, 'q_func'))
            target_q_func.restore(session, _r_iteration, os.path.join(_r_run_dir, 'target_q_func'))

    with open(os.path.join(checkpoint_dir, 'log.txt'), 'a+') as logfile:
        def print_and_log(text):
            """Print TEXT to standard output, while also writing it to the log file."""
            print(text)
            logfile.write(text + '\n')

        for t in itertools.count():
            if stopping_criterion is not None and stopping_criterion(env, t):
                break

            if type(reward_curriculum) == dict and t in reward_curriculum:
                reset_kwargs['reward_fn_name'] = reward_curriculum[t]
                print('Updated reward function to `%s`, as per the curriculum.' % reset_kwargs['reward_fn_name'])

            # Step the env and store the transition in the replay buffer
            idx = replay_buffer.store_frame(last_obs_np, dtype=_codec.obs_encoding_dtype)

            # Choose action via epsilon greedy exploration
            eps = exploration.value(t)
            if not model_initialized or random.random() < eps:
                # If first step, choose a random action
                action = env.get_random_action(**get_random_kwargs)
            else:
                obs_recent = _codec.np_to_obs(replay_buffer.encode_recent_observation())
                feed_dict = {}
                for input_name in obs_recent.keys():
                    if normalize_inputs:
                        obs_recent = copy.deepcopy(obs_recent)
                        obs_recent[input_name] = (obs_recent[input_name] - obs_mean[input_name]) / obs_std[input_name]
                    shaped_val = np.reshape(obs_recent[input_name], (1,) + obs_recent[input_name].shape)
                    feed_dict[obs_t_ph[input_name]] = shaped_val

                fetches = [q_func.outputs[output_name] for output_name in output_names]
                q_values = session.run(fetches, feed_dict=feed_dict)
                action = env.get_action_from_q_values(q_values, **get_action_kwargs)

            last_obs, reward, done = env.step(action, **step_kwargs)
            replay_buffer.store_effect(idx, action, reward, done)

            if done:
                last_obs, reward, done = env.reset(**reset_kwargs)
                last_obs_np = _codec.obs_to_np(last_obs)

                last_episode_rewards = episode_rewards
                episode_returns.append(sum(episode_rewards))
                episode_rewards = []
                play_count += 1
                game_steps = 0
            else:
                last_obs_np = _codec.obs_to_np(last_obs)
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
                obs_batch = _codec.np_to_obs(obs_batch_np, batched=True)
                next_obs_batch = _codec.np_to_obs(next_obs_batch_np, batched=True)

                if normalize_inputs:
                    if not moments_initialized:
                        # Compute observation mean and standard deviation (for use in normalization)
                        _obs_np, _, _, _, _ = replay_buffer.sample(replay_buffer.num_in_buffer - 1)
                        _obs = _codec.np_to_obs(_obs_np, batched=True)
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
                mean_episode_return = np.mean(episode_returns)
            if len(episode_returns) >= 100:
                best_mean_episode_return = max(best_mean_episode_return, mean_episode_return)
            if play_count % log_freq == 0 and game_steps == 0 and model_initialized:
                mean_return_output = 'mean return (100 episodes) %f' % mean_episode_return
                print_and_log(('step %d ' % t).ljust(len(mean_return_output), '-'))
                print_and_log(mean_return_output)
                print_and_log('best mean return (100 episodes) %f' % best_mean_episode_return)
                print_and_log('episodes %d' % play_count)
                print_and_log('exploration %f' % exploration.value(t))
                print_and_log('learning_rate %f' % optimizer_spec.lr_schedule.value(t))

                if train_params.get('log_recent_rewards', True):
                    print_and_log('mean of recent rewards %.2f' % np.mean(last_episode_rewards))
                    print_and_log('recent rewards %r' % last_episode_rewards)

                # Save network parameters
                if not evaluate_network and train_params.get('save_model', True):
                    q_func.save(session, t, outfolder=os.path.join(checkpoint_dir, 'q_func'))
                    target_q_func.save(session, t, outfolder=os.path.join(checkpoint_dir, 'target_q_func'))

                # Evaluate network
                if evaluate_network:
                    # Since `game_steps` is 0, we know that the environment has been reset
                    # We also know that `last_obs` and `last_obs_np` point to the most recent observation
                    nw_episode_rewards = []
                    while not done:
                        if save_images and len(last_obs):
                            image = env.get_image_of_state(last_obs)
                            if image is not None:
                                img_dir = os.path.join(checkpoint_dir, 'img')
                                if not os.path.exists(img_dir):
                                    os.makedirs(img_dir)
                                    print('\nSaving images to %s.' % img_dir)
                                image.save(os.path.join(img_dir, 'Game_{}_Step_{}.png'.format(play_count, game_steps)))

                        feed_dict = {}
                        for _name in input_names:
                            if normalize_inputs:
                                last_obs = copy.deepcopy(last_obs)
                                last_obs[_name] = (last_obs[_name] - obs_mean[_name]) / obs_std[_name]
                            shaped_val = np.reshape(last_obs[_name], (1,) + last_obs[_name].shape)
                            feed_dict[obs_t_ph[_name]] = shaped_val

                        fetches = [q_func.outputs[output_name] for output_name in output_names]
                        q_values = session.run(fetches, feed_dict=feed_dict)
                        action = env.get_action_from_q_values(q_values, **get_action_kwargs)
                        last_obs, reward, done = env.step(action, **step_kwargs)
                        nw_episode_rewards.append(reward)
                        game_steps += 1

                    last_obs, reward, done = env.reset(**reset_kwargs)
                    last_obs_np = _codec.obs_to_np(last_obs)
                    game_steps = 0

                    nw_return = sum(nw_episode_rewards)
                    nw_episode_returns.append(nw_return)
                    nw_mean_episode_return = np.mean(nw_episode_returns)
                    if nw_mean_episode_return > nw_best_mean_episode_return:
                        nw_best_mean_episode_return = nw_mean_episode_return
                        nw_best_iteration = t
                        if train_params.get('save_model', True):
                            q_func.save(session, t, outfolder=os.path.join(checkpoint_dir, 'q_func'))
                            target_q_func.save(session, t, outfolder=os.path.join(checkpoint_dir, 'target_q_func'))

                    print_and_log('\n--- network only ---')
                    print_and_log('return %f' % nw_return)
                    print_and_log('mean return (100 episodes) %f' % nw_mean_episode_return)
                    print_and_log('best mean return (100 episodes) %f, at step %d'
                                  % (nw_best_mean_episode_return, nw_best_iteration))

                print_and_log('')  # newline
                sys.stdout.flush()
                logfile.flush()
