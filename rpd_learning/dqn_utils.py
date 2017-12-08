#!/usr/bin/env python

"""
dqn_utils.py

A collection of utility functions useful for DQNs.
"""

import tensorflow as tf
import numpy as np
import random


def merge_dicts(*args):
    z = args[0].copy()  # start with the first dictionary's keys and values
    for y in args[1:]:
        z.update(y)  # modifies z with y's keys and values & returns None
    return z


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function SAMPLING_F that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimize OBJECTIVE using OPTIMIZER w.r.t. variables in VAR_LIST,
    while ensuring that the norm of the gradients for each variable is clipped to CLIP_VAL.
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happen if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or external precondition unsatisfied.")
        else:
            vars_left = new_vars_left


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t."""
        raise NotImplementedError('abstract method')


class PiecewiseSchedule(Schedule):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meaning that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals specified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """Value of the schedule at time t."""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(Schedule):
    def __init__(self, final_timestep, final_p, initial_p=1.0, initial_timestep=0):
        """Linear interpolation between `initial_p` and `final_p` over
        `schedule_timesteps`. After this many timesteps pass `final_p` is returned.

        Parameters
        ----------
        final_timestep: int
            Endpoint of timesteps over which to linearly anneal `initial_p` to `final_p`.
        initial_p: float
            Initial output value.
        final_p: float
            Final output value.
        initial_timestep: int
            Reference timestep for the start of the schedule. All timesteps are relative to this timestep.
        """
        self.final_timestep = final_timestep
        self.final_p = final_p
        self.initial_p = initial_p
        self.initial_timestep = initial_timestep

    def value(self, t):
        """Value of the schedule at time t."""
        fraction = min(float(t - self.initial_timestep) / (self.final_timestep - self.initial_timestep), 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory-efficient implementation of the replay buffer.

        The specific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the typical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs    = None
        self.action = None
        self.reward = None
        self.done   = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the episode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            The last `frame_history_len` frames as a single array,
            where the frames are concatenated over the final axis.
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.frame_history_len

        # If there aren't enough frames in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)

        frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
        for idx in range(start_idx, end_idx):
            frames.append(self.obs[idx % self.size])
        return np.concatenate(frames, axis=-1)

    def store_frame(self, frame, dtype='uint8'):
        """Store a single frame in the buffer at the next available index,
        overwriting old frames if necessary.

        Parameters
        ----------
        frame: np.array
            The frame to be stored.
        dtype: str
            Data type for frames (i.e. observations).

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=getattr(np, dtype))
            
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after observing frame stored at index idx.
        The reason `store_frame` and `store_effect` are broken up into two functions
        is so that one can call `encode_recent_observation` in between.

        Parameters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: np.array
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        if self.action is None:
            self.action = np.empty([self.size] + list(action.shape), dtype=np.int32)
            self.reward = np.empty([self.size],                      dtype=np.float32)
            self.done   = np.empty([self.size],                      dtype=np.bool)

        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
