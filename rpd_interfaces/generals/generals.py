#!/usr/bin/env python

"""
generals.py

Interface for generals.io. Borrows from https://github.com/toshima/generalsio.
"""

import json
import time
import threading
import collections
import numpy as np
from random import randint
from websocket import create_connection, WebSocketConnectionClosedException
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
SERVER_ENDPOINT = 'ws://botws.generals.io/socket.io/?EIO=3&transport=websocket'
REPLAY_URL_BASE = 'http://bot.generals.io/replays/'
_INVALID = 9
MODE_DEFAULT = '1v1'

_INVALID_OBS = {
    'terrain': np.full((30, 30, 1), _INVALID, np.uint8),
    'ownership': np.full((30, 30, 1), _INVALID, np.uint8),
    'armies': np.full((30, 30, 1), _INVALID, np.uint8),
    'other': np.full(34, _INVALID, np.uint8),
}

class Client(object):
    def __init__(self, endpoint):
        self._ws = create_connection(endpoint)
        self._lock = threading.RLock()

        # Start sending heartbeat
        t = threading.Thread(target=self.begin_heartbeat)
        t.daemon = True
        t.start()

    def begin_heartbeat(self):
        while True:
            try:
                with self._lock:
                    self._ws.send('2')
            except WebSocketConnectionClosedException:
                break
            time.sleep(10)

    def send(self, msg):
        """Send a message over the socket."""
        try:
            with self._lock:
                self._ws.send('42' + json.dumps(msg))
        except WebSocketConnectionClosedException:
            pass

    def receive(self):
        """Receive messages from the socket."""
        try:
            return self._ws.recv()
        except WebSocketConnectionClosedException:
            return None

    def close(self):
        """Close the socket."""
        self._ws.close()

class Generals(Environment):
    def __init__(self, user_id, reward_func=rew_total_land_dt, reward_history_cap=1000):
        self.client = Client(SERVER_ENDPOINT)
        self.user_id = user_id

        # Game data
        self.meta = {}  # playerIndex, replay_id, chat_room, team_chat_room, usernames, teams
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
        self.client.close()

    def set_username(self, username):
        """Remember: this should only be set once."""
        self.client.send(['set_username', self.user_id, username])

    def join_game(self, mode, game_id=None):
        if mode == 'private':
            if game_id is None:
                raise ValueError('game id must be provided for private games')
            self.client.send(['join_private', game_id, self.user_id])
        elif mode == '1v1':
            self.client.send(['join_1v1', self.user_id])
        elif mode == 'team':
            if game_id is None:
                raise ValueError('game id must be provided for team games')
            self.client.send(['join_team', game_id, self.user_id])
        elif mode == 'ffa':
            self.client.send(['play', self.user_id])
        else:
            raise ValueError('invalid mode')

        # Always force start
        self.client.send(['set_force_start', game_id, True])

        self.prev_mode = mode
        self.active = False
        self.active_sq = None
        print('waiting to join a %s game' % mode)

    def leave_game(self):
        """For rage quits and such."""
        self.client.send(['leave_game'])

    def reset(self, mode=None):
        """Reset game. This just means starting a new one."""
        if mode is None:
            mode = self.prev_mode or MODE_DEFAULT
        self.join_game(mode)

        # Wait until we have our first observation, then return it
        return self.wait_for_next_observation()

    def get_updates(self):
        while True:
            msg = self.client.receive()
            if msg is None or not msg.strip():
                break
            if msg in {'3', '40'}:  # heartbeats, connection ACKs
                continue

            # Remove numeric prefix
            while msg and msg[0].isdigit():
                msg = msg[1:]

            # Load message
            msg = json.loads(msg)
            if not isinstance(msg, list):
                continue

            if msg[0] == 'error_user_id':
                raise ValueError('already in game')
            elif msg[0] == 'pre_game_start':
                print('game is about to start')
            elif msg[0] == 'game_start':
                self.meta = msg[1]
                self.player_index = self.meta['playerIndex']
                replay_url = REPLAY_URL_BASE + self.meta['replay_id']
                print('Game starting! The replay will be available after the game at %s' % replay_url)
            elif msg[0] == 'game_update':
                # Contents: turn, map_diff, cities_diff, generals, scores, stars
                yield self.process_update(msg[1])
            elif msg[0] in {'game_won', 'game_lost'}:
                yield self.process_update(msg[1], result=msg[0])
                break
            else:
                print('unknown message type: {}'.format(msg))

    def process_update(self, data, result=None):
        """Process and return a game update."""
        if result is not None:
            return {'result': result}

        self.active = True
        if self.active_sq is None:
            self.general_sq = data['generals'][self.player_index]
            self.active_sq = self.general_sq
            print('[i] general square: %d' % self.general_sq)

        self.cities = self.patch(self.cities, data['cities_diff'])  # currently visible cities
        self.map = self.patch(self.map, data['map_diff'])  # current map state (dimensions, armies, terrain)

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
    def patch(old, diff):
        out, i = [], 0
        while i < len(diff):
            out.extend(old[len(out):len(out)+diff[i]])
            i += 1
            if i < len(diff):
                out.extend(diff[i+1:i+1+diff[i]])
                i += diff[i]
            i += 1
        return out

    @staticmethod
    def _pad(lst, val, size):
        """Destructively right-pad a list with VAL until its length is equal to SIZE.
        Return said list.
        """
        lst.extend([val] * (size - len(lst)))
        return lst

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
