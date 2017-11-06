#!/usr/bin/env python

"""
dqn.py

Deep Q-network as described by CS 294-112 (goo.gl/MhA4eA).
"""

import sys
import itertools
import tensorflow as tf
from collections import namedtuple

from rpd_learning.dqn_utils import *
from rpd_learning.models import Model

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

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
    input_shape = arch['inputs']['observation']['shape']
    action_dim = arch['outputs']['action']['shape'][0]

    # Set up placeholders
    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))  # current observation (or state)
    act_t_ph = tf.placeholder(tf.int32, [None])  # current action
    rew_t_ph = tf.placeholder(tf.float32, [None])  # current reward
    obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))  # next observation (or state)
    done_mask_ph = tf.placeholder(tf.float32, [None])  # end of episode mask (1 if next state = end of an episode)

    # Casting to float on GPU ensures lower data transfer times
    obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # Create networks (for current/next Q-values)
    q_func = Model(arch, inputs={'observation': obs_t_ph}, scope='q_func', reuse=False)  # model to use for computing the q-function
    target_q_func = Model(arch, inputs={'observation': obs_tp1_float}, scope='target_q_func', reuse=False)
    q_func_out = q_func.outputs['action']
    target_out = target_q_func.outputs['action']

    # Compute the Bellman error
    q_j = tf.reduce_sum(tf.multiply(tf.one_hot(act_t_ph, action_dim), q_func_out), axis=1)
    y_j = rew_t_ph + tf.multiply(gamma, tf.reduce_max(tf.stop_gradient(target_out), axis=1))
    total_error = tf.reduce_mean(tf.square(y_j - q_j))  # scalar valued tensor representing Bellman error (evaluate the current and next Q-values and construct corresponding error)
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
    last_obs = env.reset()
    log_freq = 10000

    for t in itertools.count():
        # 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # 2. Step the env and store the transition in the replay buffer
        idx = replay_buffer.store_frame(last_obs)
        if not model_initialized:
            # If first step, choose a random action
            action = env.get_random_action()
        else:
            # Choose action via epsilon greedy exploration
            eps = exploration.value(t)
            obs_recent = replay_buffer.encode_recent_observation()
            q_values = session.run(q_func_out, feed_dict={
                obs_t_ph: np.reshape(obs_recent, (1,) + obs_recent.shape),
            })
            probabilities = np.full(action_dim, float(eps) / (action_dim - 1))
            probabilities[np.argmax(q_values)] = 1.0 - eps
            action = np.random.choice(action_dim, p=probabilities)

        last_obs, reward, done = env.apply_action(action)
        replay_buffer.store_effect(idx, action, reward, done)
        if done:
            last_obs = env.reset()

        # At this point, the environment should have been advanced one step (and
        # reset if done was true), last_obs should point to the new latest observation,
        # and the replay buffer should contain one more transition.

        # 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size):
            # Use replay buffer to sample a batch of transitions
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            # Initialize the model
            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), {
                    obs_t_ph: obs_batch,
                    obs_tp1_ph: next_obs_batch,
                })
                model_initialized = True

            # Train the model
            session.run(train_fn, feed_dict={
                obs_t_ph: obs_batch,
                act_t_ph: act_batch,
                rew_t_ph: rew_batch,
                obs_tp1_ph: next_obs_batch,
                done_mask_ph: done_mask,
                learning_rate: optimizer_spec.lr_schedule.value(t),
            })

            # Periodically update the target network
            if num_param_updates % target_update_freq == 0:
                session.run(update_target_fn)
            num_param_updates += 1

        # 4. Log progress
        episode_rewards = env.reward_history  # TODO separate rewards into episodes
        # if len(episode_rewards) > 0:
        #     mean_episode_reward = np.mean(episode_rewards[-100:])
        # if len(episode_rewards) > 100:
        #     best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % log_freq == 0 and model_initialized:
            print('timestep %d' % t)
            print('mean reward %.2f' % np.mean(episode_rewards))
            print('recent rewards %r' % episode_rewards[-5:])
            # print("mean reward (100 episodes) %f" % mean_episode_reward)
            # print("best mean reward %f" % best_mean_episode_reward)
            # print("episodes %d" % len(episode_rewards))
            print('exploration %f' % exploration.value(t))
            print('learning_rate %f' % optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()
