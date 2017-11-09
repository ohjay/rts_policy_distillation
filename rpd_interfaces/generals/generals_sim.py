#!/usr/bin/env python

"""
generals_sim.py

Interface for the simulated version of generals.io.
This version is local and doesn't require any networking.
"""

import copy
import collections
import numpy as np
from random import randint
import rpd_interfaces.generals.simulator as sim
from rpd_interfaces.interfaces import Environment
from rpd_interfaces.generals.reward import rew_total_land_dt

# Terrain constants
# -----------------
TILE_EMPTY = -1
TILE_MOUNTAIN = -2
TILE_FOG = -3
TILE_FOG_OBSTACLE = -4  # cities and mountains show up as obstacles in the fog of war
TILE_CITY = -5

# Action constants
# ----------------
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NOOP = 4
ACTION_UP_50 = 5
ACTION_DOWN_50 = 6
ACTION_LEFT_50 = 7
ACTION_RIGHT_50 = 8

# General constants
# -----------------
_INVALID = 9

_INVALID_OBS = {
    'terrain': np.full((30, 30, 1), _INVALID, np.uint8),
    'ownership': np.full((30, 30, 1), _INVALID, np.uint8),
    'armies': np.full((30, 30, 1), _INVALID, np.uint8),
}

class GeneralsSim(Environment):
    def __init__(self, map_shape, map_player_count, reward_func=rew_total_land_dt, reward_history_cap=1000):
        # Game data
        self.height, self.width = map_shape
        self.player_count = map_player_count
        self.map = None

        self.player_id = -1
        self.active_sq = None

        # Learning-related info
        self.prev_observation = None
        self.reward_func = reward_func
        self.reward_history = collections.deque(maxlen=reward_history_cap)

    def reset(self):
        """Reset game and return the first observation."""
        self.map = sim.Map((self.height, self.width), self.player_count)
        self.player_id = 1  # TODO: choose player ID for specific DQN? self-play?
        self.active_sq = self.map.players[self.player_id].general_loc
        obs = self.map.generate_state(self.map.players[self.player_id])
        return self._reformat_observation(obs)

    def _parse_action(self, action):
        """Obtains the (start_loc, end_loc) encoded by ACTION."""
        end_loc = copy.copy(self.active_sq)
        if action == ACTION_UP:
            end_loc[0] -= 1
        elif action == ACTION_DOWN:
            end_loc[0] += 1
        elif action == ACTION_LEFT:
            end_loc[1] -= 1
        elif action == ACTION_RIGHT:
            end_loc[1] += 1

        return self.active_sq, end_loc

    def get_random_action(self):
        """Get a random move."""
        return randint(0, 4)

    def apply_action(self, action, random=False):
        """Actions are formatted as integers from 0-4 representing the choice of move (up, down, left, right, noop).
        If the action is invalid, nothing will be done.

        Returns an (observation, reward, done) tuple that represents the result of taking the action.
        """
        start_loc, end_loc = self._parse_action(action)
        next_obs, _, done = self.map.action(self.player_id, start_loc, end_loc)  # our simulator handles noops
        next_obs = self._reformat_observation(next_obs)

        obs, self.prev_observation = self.prev_observation, next_obs
        reward = self.reward_func(obs, action, next_obs, self)
        self.reward_history.append(reward)

        return next_obs, reward, done

    @staticmethod
    def _pad2d(arr, val, des_shape):
        """Pad a 2D array with VAL s.t. its final dimensions are (DES_SHAPE[0], DES_SHAPE[1])."""
        final = np.full(des_shape, val)
        final[:arr.shape[0], :arr.shape[1]] = arr
        return final

    def _reformat_observation(self, obs):
        """Reformats a map-generated observation [a (terrain, armies, ownership) tuple]
        as a dictionary of NumPy arrays with the following keys and values:

        - 'terrain' [shape (30, 30, 1)]: the terrain map
        - 'ownership' [shape (30, 30, 1)]: the ownership map
        - 'armies' [shape (30, 30, 1)]: the army map
        - 'other' [shape 34]:
          - index 0: width of the map
          - index 1: turn number
          - indices 2-9: general positions (-1 indicates "unknown")
          - indices 10-33: (# tiles, # units, 0 if dead else 1) for each player, ordered by index
        """
        observation = {
            'terrain': self._pad2d(obs[0], _INVALID, (30, 30))[:, :, None],
            'armies': self._pad2d(obs[1], _INVALID, (30, 30))[:, :, None],
            'ownership': self._pad2d(obs[2], _INVALID, (30, 30))[:, :, None]
        }

        # TODO: get additional game state ("other") as per the official generals.io interface (?)
        # for transfer to "real"

        return observation
