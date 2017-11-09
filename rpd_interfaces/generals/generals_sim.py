#!/usr/bin/env python

"""
generals_sim.py

Interface for the simulated version of generals.io.
This version is local and doesn't require any networking.
"""

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

# General constants
# -----------------
_INVALID = 9

_INVALID_OBS = {
    'terrain': np.full((30, 30, 1), _INVALID, np.uint8),
    'ownership': np.full((30, 30, 1), _INVALID, np.uint8),
    'armies': np.full((30, 30, 1), _INVALID, np.uint8),
    'other': np.full(34, _INVALID, np.uint8),
}

# TODO: this entire file is in progress

class GeneralsSim(Environment):
    def __init__(self, map_shape, map_player_count, reward_func=rew_total_land_dt, reward_history_cap=1000):
        # Game data
        self.height, self.width = map_shape
        self.player_count = map_player_count
        self.map = None

        self.player_index = -1
        self.general_sq = -1
        self.cities = []
        self.map = []
        self.width, self.height = -1, -1
        self.armies = None
        self.terrain = None
        self.stars = []
        self.active = False
        self.move_id = 1
        self.prev_mode = None
        self.active_sq = None
        self.turn = -1
        self.turn_throttle = -1

        # Learning-related info
        self.prev_observation = None
        self.reward_func = reward_func
        self.reward_history = collections.deque(maxlen=reward_history_cap)

    def close(self):
        """Close and clean up the environment."""
        self.map = None

    def reset(self):
        """Reset game and return the first observation."""
        self.map = sim.Map((self.height, self.width), self.player_count)
        return 'AN OBSERVATION'  # TODO

    def process_update(self, data, result=None):
        """Process and return a game update."""
        if result is not None:
            return {'result': result}

        self.active = True
        if self.active_sq is None:
            self.general_sq = data['generals'][self.player_index]
            self.active_sq = self.general_sq
            print('[i] general square: %d' % self.general_sq)

        generals = data['generals']
        self.width, self.height = self.map[0], self.map[1]
        size = self.width * self.height

        # Extract army and terrain values
        self.armies = self.map[2:size+2]
        self.terrain = self.map[size+2:size+2+size]

        self.turn = data['turn']
        scores = data['scores']

        if 'stars' in data:
            self.stars[:] = data['stars']

        return {
            'result': result,
            'generals': generals,  # array of general positions ordered by index (-1 indicates "unknown")
            'width': self.width,
            'height': self.height,
            'size': size,
            'armies': self.armies,  # quantities of army units for each square
            'terrain': self.terrain,  # type of each square (-4, -3, -2, -1, or a player index indicating ownership)
            'turn': self.turn,
            'scores': scores,  # list of dictionaries containing score information for each player
            'cities': self.cities,
        }

    def wait_for_next_observation(self):
        for update in self.get_updates():
            if type(update) == dict and update['result'] is None:
                return self.extract_observation(update)
        return _INVALID_OBS  # done

    @staticmethod
    def _pad2d(arr, val, des_shape):
        """Pad a 2D array with VAL s.t. its final dimensions are (DES_SHAPE[0], DES_SHAPE[1])."""
        final = np.full(des_shape, val)
        final[:arr.shape[0], :arr.shape[1]] = arr
        return final

    def extract_observation(self, update):
        """Returns an observation as a tensor.

        For generals.io, an observation is a dictionary of NumPy arrays, structured as follows:
        - 'terrain' [shape (30, 30, 1)]: the terrain map
        - 'ownership' [shape (30, 30, 1)]: the ownership map
        - 'armies' [shape (30, 30, 1)]: the army map
        - 'other' [shape 34]:
          - index 0: width of the map
          - index 1: turn number
          - indices 2-9: general positions (-1 indicates "unknown")
          - indices 10-33: (# tiles, # units, 0 if dead else 1) for each of eight potential players, ordered by index

        Whenever applicable, -2 indicates "nonexistent / invalid".
        Maximum capacity is eight players and a 30x30 map.
        """
        if update['result'] is not None:
            # TODO this needs to represent a high-reward observation if we win, and vice-versa if we lose
            return _INVALID_OBS

        _scores = []
        for s in update['scores']:
            _scores.extend([s['tiles'], s['total'], 1 - int(s['dead'])])

        terrain = update['terrain']
        ownership = np.reshape([max(-1, id) for id in update['terrain']], (self.height, self.width))
        for city_index in update['cities']:
            terrain[city_index] = TILE_CITY
        terrain = np.reshape(terrain, (self.height, self.width))
        armies = np.reshape(update['armies'], (self.height, self.width))

        observation = {
            'terrain': self._pad2d(terrain, _INVALID, (30, 30))[:, :, None],
            'ownership': self._pad2d(ownership, _INVALID, (30, 30))[:, :, None],
            'armies': self._pad2d(armies, _INVALID, (30, 30))[:, :, None],
            'other': np.array(
                [update['width']] +
                [update['turn']] +
                self._pad(update['generals'], _INVALID, 8) +
                self._pad(_scores, _INVALID, 24)
            ).astype(np.float32),
        }
        self.prev_observation = observation
        return observation

    def attack(self, start_index, end_index, is_half=False):
        if not self.active:
            raise ValueError('no updates have been received')
        if self.turn > self.turn_throttle:
            self.active_sq = end_index
            if self._valid(start_index, end_index):
                self.client.send(['attack', start_index, end_index, is_half, self.move_id])
                self.move_id += 1
            self.turn_throttle = self.turn

    def _parse_action(self, action):
        """Obtains the (start_index, end_index) encoded by ACTION."""
        # start_index, end_index = int(round(action[0])), int(round(action[1]))
        # start_index, end_index = int(action) // 1000, int(action) % 1000

        end_index = None
        if action == ACTION_UP:
            end_index = self.active_sq - self.width
        elif action == ACTION_DOWN:
            end_index = self.active_sq + self.width
        elif action == ACTION_LEFT:
            end_index = self.active_sq - 1
        elif action == ACTION_RIGHT:
            end_index = self.active_sq + 1

        return self.active_sq, end_index

    def _valid(self, start_index, end_index):
        """Returns True if the given action is currently valid."""
        if self.terrain[start_index] != self.player_index or self.armies[start_index] < 2:
            return False
        if self.terrain[end_index] in (TILE_MOUNTAIN, TILE_FOG_OBSTACLE):
            return False
        row = start_index // self.width
        col = start_index % self.width
        if end_index == start_index - 1:
            return col > 0
        if end_index == start_index + 1:
            return col < self.width - 1
        if end_index == start_index + self.width:
            return row < self.height - 1
        if end_index == start_index - self.width:
            return row > 0
        return False

    def get_random_action(self):
        """Get a random move."""
        return randint(0, 4)

    def apply_action(self, action, random=False):
        """Actions are formatted as integers from 0-4 representing the choice of move (up, down, left, right, noop).
        If the action is invalid, nothing will be done.

        Returns an (observation, reward, done) tuple.

        TODO: potential representations
        (1) (int, int) from/to tuple (when applying, can round to nearest)
        (2) (900 + 900,) array of 0s and 1s (when applying, can use argmax for each of from/to ranges

        TODO: incorporate 50% moves
        """
        # if not self._valid(action):
        #     print('invalid action, choosing one at random')
        #     action = self.get_random_action()

        start_index, end_index = self._parse_action(action)
        if end_index is not None:
            self.attack(start_index, end_index)

        # Generate return info
        obs = self.prev_observation
        next_obs = self.wait_for_next_observation()
        reward = self.reward_func(obs, action, next_obs, self)
        done = np.count_nonzero(next_obs['other'] - _INVALID) == 0

        self.reward_history.append(reward)
        return next_obs, reward, done
