#!/usr/bin/env python

"""
obs_codecs.py

Encoders and decoders for environment-generated observations.
"""

import operator
import functools
import numpy as np

from rpd_learning.general_utils import dominant_dtype


class StandardCodec(object):
    def __init__(self, input_names, input_shapes, input_dtypes):
        self.input_names = input_names
        self.input_shapes = input_shapes  # should have the same order as `input_shapes`

        # Information to be filled in about the encoding, so we don't have to determine it every time
        self.obs_encoding = {}
        self.obs_encoding_dtype = dominant_dtype(input_dtypes)

    def obs_to_np(self, obs):
        """Reformats observation as a single NumPy array.
        An observation, at least from the RPD interface, will be given as an {input_name: value} dict.
        """
        if 'encoding_option' in self.obs_encoding:
            # Expedited route (don't need to run as many condition tests)
            encoding_option = self.obs_encoding['encoding_option']
            if encoding_option == 1:
                return obs[self.input_names[0]]
            elif encoding_option == 2:
                return np.stack([obs[input_name] for input_name in self.input_names], axis=0)
            elif encoding_option == 3:
                return np.concatenate([obs[input_name] for input_name in self.input_names],
                                      axis=self.obs_encoding['axis'])
            elif encoding_option == 4:
                return np.concatenate([obs[input_name].flatten() for input_name in self.input_names])

        if len(self.input_names) == 1:
            self.obs_encoding['encoding_option'] = 1
            return obs[self.input_names[0]]  # if there is only one input, return as-is (1)

        num_dims0 = len(self.input_shapes[0])
        if all(len(_shape) == num_dims0 for _shape in self.input_shapes[1:]):
            # If it's possible to join the inputs along a new axis, do it (2)
            shape0 = self.input_shapes[0]
            if all(_shape == shape0 for _shape in self.input_shapes[1:]):
                self.obs_encoding['encoding_option'] = 2
                return np.stack([obs[input_name] for input_name in self.input_names], axis=0)

            # If it's possible to join the inputs along an existing axis, do it (3)
            for dim in range(num_dims0):
                without_axis0 = shape0[:dim] + shape0[dim + 1:]
                if all(_shape[:dim] + _shape[dim + 1:] == without_axis0 for _shape in self.input_shapes[1:]):
                    self.obs_encoding['encoding_option'] = 3
                    self.obs_encoding['axis'] = dim
                    return np.concatenate([obs[input_name] for input_name in self.input_names], axis=dim)

        # Return a concatenation of flattened arrays (4)
        self.obs_encoding['encoding_option'] = 4
        return np.concatenate([obs[input_name].flatten() for input_name in self.input_names])

    def np_to_obs(self, obs_np, batched=False):
        """Separates observation into individual inputs (the {input_name: value} dict it was originally).
        Note: it is possible that OBS_NP represents not a single observation, but an entire batch of observations.

        This the inverse of `obs_to_np`.
        """
        if 'encoding_option' in self.obs_encoding:
            encoding_option = self.obs_encoding['encoding_option']
            if encoding_option == 1:
                return {self.input_names[0]: obs_np}
            elif encoding_option == 2:
                axis = 1 if batched else 0
                individual = np.split(obs_np, obs_np.shape[axis], axis=axis)
                return {input_name: individual[i] for i, input_name in enumerate(self.input_names)}
            elif encoding_option == 3:
                split_indices = np.cumsum([_shape[self.obs_encoding['axis']] for _shape in self.input_shapes])
                axis = self.obs_encoding['axis'] + 1 if batched else self.obs_encoding['axis']
                individual = np.split(obs_np, split_indices, axis=axis)
                return {input_name: individual[i] for i, input_name in enumerate(self.input_names)}
                # Encoding option #4 shall be handled below
        else:
            if len(self.input_names) == 1:
                self.obs_encoding['encoding_option'] = 1
                return {self.input_names[0]: obs_np}  # inverse of (1)

            num_dims0 = len(self.input_shapes[0])
            if all(len(_shape) == num_dims0 for _shape in self.input_shapes[1:]):
                # Inverse of (2)
                shape0 = self.input_shapes[0]
                if all(_shape == shape0 for _shape in self.input_shapes[1:]):
                    self.obs_encoding['encoding_option'] = 2
                    axis = 1 if batched else 0
                    individual = np.split(obs_np, obs_np.shape[axis], axis=axis)
                    return {input_name: individual[i] for i, input_name in enumerate(self.input_names)}

                # Inverse of (3)
                for dim in range(num_dims0):
                    without_axis0 = shape0[:dim] + shape0[dim + 1:]
                    if all(_shape[:dim] + _shape[dim + 1:] == without_axis0 for _shape in self.input_shapes[1:]):
                        self.obs_encoding['encoding_option'] = 3
                        self.obs_encoding['axis'] = dim
                        split_indices = np.cumsum([_shape[dim] for _shape in self.input_shapes])
                        axis = dim + 1 if batched else dim
                        individual = np.split(obs_np, split_indices, axis=axis)
                        return {input_name: individual[i] for i, input_name in enumerate(self.input_names)}

        # Inverse of (4)
        self.obs_encoding['encoding_option'] = 4
        obs, i = {}, 0
        for input_name, _shape in zip(self.input_names, self.input_shapes):
            size = functools.reduce(operator.mul, _shape, 1)
            if batched:
                _shape = np.insert(_shape, 0, obs_np.shape[0])
                try:
                    obs[input_name] = np.reshape(obs_np[:, i:i + size], _shape)
                except ValueError:
                    print('i: %d' % i)
                    print('obs_np shape: %r' % (obs_np.shape,))
                    print('shape, size: %r, %d' % (_shape, size))
                    raise
            else:
                obs[input_name] = np.reshape(obs_np[i:i + size], _shape)
            i += size
        return obs
