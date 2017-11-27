#!/usr/bin/env python

"""
dqn.py

Deep Q-network as described by CS 294-112 (goo.gl/MhA4eA).
"""

import os, sys
import operator
import datetime
import functools, itertools
import tensorflow as tf
from collections import namedtuple

from rpd_learning.dqn_utils import *
from rpd_learning.models import Model
from random import random


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

_LAUNCH_TIME = datetime.datetime.now()
_NO_OP = 4

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
    exploration: dqn_utils.Schedule
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

    ###############
    # BUILD MODEL #
    ###############

    arch = config['dqn_arch']
    num_dir = arch['outputs']['direction']['shape'][0]
    num_x = arch['outputs']['target_x']['shape'][0]
    num_y = arch['outputs']['target_y']['shape'][0]


    def _obs_to_np(obs):
        """Reformats observation as a single NumPy array.
        An observation, at least from the RPD interface, will be given as an {input_name: value} dict.
        """
        return np.concatenate([obs[input_name].flatten() for input_name in sorted(arch['inputs'].keys())])

    def _np_to_obs(obs_np):
        """Separates observation into individual inputs (the {input_name: value} dict it was originally).
        This the inverse of `_obs_to_np`.
        """
        obs = {}
        i = 0
        for input_name in sorted(arch['inputs'].keys()):
            shape = arch['inputs'][input_name]['shape']
            size = functools.reduce(operator.mul, shape, 1)
            if obs_np.shape[0] == batch_size:
                shape = np.insert(shape, 0, obs_np.shape[0])
                obs[input_name] = np.reshape(obs_np[:, i:i+size], shape)
            else:
                obs[input_name] = np.reshape(obs_np[i:i+size], shape)
            i += size
        return obs

    # Set up placeholders
    obs_t_ph = {input_name: None for input_name in arch['inputs'].keys()}  # current observation (or state)
    obs_t_ph_float = {input_name: None for input_name in arch['inputs'].keys()}
    for input_name in obs_t_ph:
        info = arch['inputs'][input_name]
        dtype, shape = info['dtype'], info['shape']
        obs_t_ph[input_name] = tf.placeholder(getattr(tf, dtype), [None] + list(shape))
        if dtype == 'float32':
            obs_t_ph_float[input_name] = obs_t_ph[input_name]
        else:  # casting to float on GPU ensures lower data transfer times
            obs_t_ph_float[input_name] = tf.cast(obs_t_ph[input_name], tf.float32) / 255.0
    act_t_x_ph = tf.placeholder(tf.int32, [None]) # current action
    act_t_y_ph = tf.placeholder(tf.int32, [None])
    act_t_dir_ph = tf.placeholder(tf.int32, [None])  
    rew_t_ph = tf.placeholder(tf.float32, [None])  # current reward
    obs_tp1_ph = {input_name: None for input_name in arch['inputs'].keys()}  # next observation (or state)
    obs_tp1_ph_float = {input_name: None for input_name in arch['inputs'].keys()}
    for input_name in obs_tp1_ph:
        info = arch['inputs'][input_name]
        dtype, shape = info['dtype'], info['shape']
        obs_tp1_ph[input_name] = tf.placeholder(getattr(tf, dtype), [None] + list(shape))
        if dtype == 'float32':
            obs_tp1_ph_float[input_name] = obs_tp1_ph[input_name]
        else:
            obs_tp1_ph_float[input_name] = tf.cast(obs_tp1_ph[input_name], tf.float32) / 255.0
    done_mask_ph = tf.placeholder(tf.float32, [None])  # end of episode mask (1 if next state = end of an episode)

    # Create networks (for current/next Q-values)
    q_func = Model(arch, inputs=obs_t_ph_float, scope='q_func', reuse=False)  # model to use for computing the q-function
    target_q_func = Model(arch, inputs=obs_tp1_ph_float, scope='target_q_func', reuse=False)
    
    q_dir_out = q_func.outputs['direction']
    q_x_out = q_func.outputs['target_x']
    q_y_out = q_func.outputs['target_y']
    target_dir_out = target_q_func.outputs['direction']
    target_x_out = target_q_func.outputs['target_x']
    target_y_out = target_q_func.outputs['target_y']

    # Compute the Bellman error
    q_x_j = tf.reduce_sum(tf.multiply(tf.one_hot(act_t_x_ph, num_x), q_x_out), axis=1)
    y_x_j = rew_t_ph + tf.multiply(gamma, tf.reduce_max(tf.stop_gradient(target_x_out), axis=1))
    q_y_j = tf.reduce_sum(tf.multiply(tf.one_hot(act_t_y_ph, num_y), q_y_out), axis=1)
    y_y_j = rew_t_ph + tf.multiply(gamma, tf.reduce_max(tf.stop_gradient(target_y_out), axis=1))
    q_dir_j = tf.reduce_sum(tf.multiply(tf.one_hot(act_t_dir_ph, num_dir), q_dir_out), axis=1)
    y_dir_j = rew_t_ph + tf.multiply(gamma, tf.reduce_max(tf.stop_gradient(target_dir_out), axis=1))

    total_error = tf.reduce_mean(tf.square(y_x_j - q_x_j)) + \
                  tf.reduce_mean(tf.square(y_y_j - q_y_j)) + \
                  tf.reduce_mean(tf.square(y_dir_j - q_dir_j)) # scalar valued tensor representing Bellman error (evaluate the current and next Q-values and construct corresponding error)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')  # all vars in Q-function network
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')  # all vars in target network

    # Construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
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

    model_initialized = False
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    player_output, _ = env.reset(map_init='empty')
    state, reward, done = player_output
    last_obs_np = _obs_to_np(state)
    log_freq = 150
    play_count = 0
    game_steps = 0

    last_episode_rewards = []
    episode_rewards = []

    save_images = False

    for t in itertools.count():
        # 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # 2. Step the env and store the transition in the replay buffer
        idx = replay_buffer.store_frame(last_obs_np)
        eps = exploration.value(t)
        if not model_initialized or random() < eps:
            # If first step, choose a random action
            action = env.get_random_semi_valid_action(1)
        else:
            # Choose action via epsilon greedy exploration
            obs_recent = _np_to_obs(replay_buffer.encode_recent_observation())
            feed_dict = {}
            for input_name in obs_recent.keys():
                shaped_val = np.reshape(obs_recent[input_name], (1,) + obs_recent[input_name].shape)
                feed_dict[obs_t_ph[input_name]] = shaped_val
            q_x = session.run(q_x_out, feed_dict=feed_dict)
            q_y = session.run(q_y_out, feed_dict=feed_dict)
            q_dir = session.run(q_dir_out, feed_dict=feed_dict)

            act_x, act_y, act_dir = np.argmax(q_x), np.argmax(q_y), np.argmax(q_dir)
            action = np.array((act_x, act_y, act_dir))

        player_output, _ = env.step(action)
        last_obs, reward, done = player_output
        replay_buffer.store_effect(idx, action, reward, done)
        if save_images and len(last_obs):
            run_dir = os.path.join('img', 'run_%s' % _LAUNCH_TIME.strftime('%m-%d__%H_%M'))
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
                print('Created directory at %s.' % run_dir)
            env.get_image_of_state(last_obs).save("{}/Game_{}_Step_{}.png".format(run_dir, play_count, game_steps))

        if done:
            player_output, _ = env.reset(map_init='empty')
            last_obs, reward, done = player_output
            last_obs_np = _obs_to_np(last_obs)

            last_episode_rewards = episode_rewards
            episode_rewards = []
            save_images = False
            play_count += 1
            game_steps = 0
        else:
            last_obs_np = _obs_to_np(last_obs)
            episode_rewards.append(reward)
            game_steps += 1

        # At this point, the environment should have been advanced one step (and
        # reset if done was true), last_obs_np should point to the new latest observation,
        # and the replay buffer should contain one more transition.

        # 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size):
            # Use replay buffer to sample a batch of transitions
            obs_batch_np, act_batch, rew_batch, next_obs_batch_np, done_mask = replay_buffer.sample(batch_size)
            obs_batch = _np_to_obs(obs_batch_np)
            next_obs_batch = _np_to_obs(next_obs_batch_np)
            # Initialize the model
            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), merge_dicts(
                    {obs_t_ph[input_name]: obs_batch[input_name] for input_name in obs_batch.keys()},
                    {obs_tp1_ph[input_name]: next_obs_batch[input_name] for input_name in next_obs_batch.keys()}
                ))
                model_initialized = True

            # Train the model
            session.run(train_fn, merge_dicts(
                {obs_t_ph[input_name]: obs_batch[input_name] for input_name in obs_batch.keys()},
                {obs_tp1_ph[input_name]: next_obs_batch[input_name] for input_name in next_obs_batch.keys()},
                {act_t_x_ph: act_batch[:,0], act_t_y_ph: act_batch[:,1], act_t_dir_ph: act_batch[:,2],  
                 rew_t_ph: rew_batch, done_mask_ph: done_mask, learning_rate: optimizer_spec.lr_schedule.value(t)}
            ))

            # Periodically update the target network
            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)
            num_param_updates += 1

        # 4. Log progress
        # last_episode_rewards = env.reward_history  # TODO separate rewards into episodes
        # if len(last_episode_rewards) > 0:
        #     mean_episode_reward = np.mean(last_episode_rewards[-100:])
        # if len(last_episode_rewards) > 100:
        #     best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if play_count % log_freq == 0 and game_steps == 0 and model_initialized:
            print('timestep %d' % t)
            print('mean reward %.2f' % np.mean(last_episode_rewards))
            print('recent rewards %r' % last_episode_rewards)
            # print("mean reward (100 episodes) %f" % mean_episode_reward)
            # print("best mean reward %f" % best_mean_episode_reward)
            # print("episodes %d" % len(last_episode_rewards))
            print('exploration %f' % exploration.value(t))
            print('learning_rate %f' % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()

            # Save network parameters
            q_func.save(session, t, outfolder='q_func')
            target_q_func.save(session, t, outfolder='target')  # maybe don't need to do both
            save_images = True
