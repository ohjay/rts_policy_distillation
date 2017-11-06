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
from random import randint, random
from websocket import create_connection, WebSocketConnectionClosedException
from rpd_interfaces.interfaces import Environment
from rpd_interfaces.generals.reward import rew_total_land_dt

# Terrain constants
# -----------------
TILE_EMPTY = -1
TILE_MOUNTAIN = -2
TILE_FOG = -3
TILE_FOG_OBSTACLE = -4  # cities and mountains show up as obstacles in the fog of war

# General constants
# -----------------
SERVER_ENDPOINT = 'ws://botws.generals.io/socket.io/?EIO=3&transport=websocket'
REPLAY_URL_BASE = 'http://bot.generals.io/replays/'
_INVALID = 2
MODE_DEFAULT = '1v1'

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
        self.cities = []
        self.map = []
        self.terrain = None
        self.stars = []
        self.active = False
        self.move_id = 1
        self.prev_mode = None

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
                yield self.process_update(msg[0], result=msg[1])
                break
            else:
                print('unknown message type: {}'.format(msg))

    def process_update(self, data, result=None):
        """Process and return a game update."""
        if result is not None:
            return {'result': result}

        self.active = True

        self.cities = self.patch(self.cities, data['cities_diff'])  # currently visible cities
        self.map = self.patch(self.map, data['map_diff'])  # current map state (dimensions, armies, terrain)

        generals = data['generals']
        width, height = self.map[0], self.map[1]
        size = width * height

        # Extract army and terrain values
        armies = self.map[2:size+2]
        self.terrain = self.map[size+2:size+2+size]

        turn = data['turn']
        scores = data['scores']

        if 'stars' in data:
            self.stars[:] = data['stars']

        return {
            'result': result,
            'generals': generals,  # array of general positions ordered by index (-1 indicates "unknown")
            'width': width,
            'height': height,
            'size': size,
            'armies': armies,  # quantities of army units for each square
            'terrain': self.terrain,  # type of each square (-4, -3, -2, -1, or a player index indicating ownership)
            'turn': turn,
            'scores': scores,  # list of dictionaries containing score information for each player
            'cities': self.cities,
        }

    def wait_for_next_observation(self):
        for update in self.get_updates():
            if type(update) == dict and update['result'] is None:
                return self.extract_observation(update)

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

    def extract_observation(self, update):
        """Returns an observation as a tensor.

        For generals.io, an observation is a (1884,) NumPy array
        with the following contents at the following indices:

        - index 0: width of the map
        - index 1: turn number
        - indices 2-9: general positions (-1 indicates "unknown")
        - indices 10-33: (# tiles, # units, 0 if dead else 1) for each of eight potential players, ordered by index
        - indices 34-83: ordered list of city indices
        - indices 84-983: terrain info for each square, encoded as an integer from -4 to 7
        - indices 984-1883: quantities of army units for each square index

        Whenever applicable, -2 indicates "nonexistent / invalid".
        Maximum capacity is eight players and a 30x30 map.
        """
        _scores = []
        for s in update['scores']:
            _scores.extend([s['tiles'], s['total'], 1 - int(s['dead'])])
        observation = np.array(
            [update['width']] +
            [update['turn']] +
            self._pad(update['generals'], _INVALID, 8) +
            self._pad(_scores, _INVALID, 24) +
            self._pad(update['cities'], _INVALID, 30) +
            self._pad(update['terrain'], _INVALID, 900) +
            self._pad(update['armies'], _INVALID, 900)
        ).astype(np.int16)  # the only things outside of byte range are scores
        self.prev_observation = observation
        return observation

    def attack(self, start_index, end_index, is_half=False):
        if not self.active:
            raise ValueError('no updates have been received')
        self.client.send(['attack', start_index, end_index, is_half, self.move_id])
        self.move_id += 1

    def _valid(self, action):
        """Returns True if the given action is currently valid."""
        start_index, end_index = int(round(action[0])), int(round(action[1]))
        if self.terrain[start_index] != self.player_index:
            return False
        width, height = self.map[0], self.map[1]
        row = start_index // width
        col = start_index % width
        if end_index == start_index - 1:
            return col > 0
        if end_index == start_index + 1:
            return col < width - 1
        if end_index == start_index + width:
            return row < height - 1
        if end_index == start_index - width:
            return row > 0
        return False

    def get_random_action(self):
        """Get a random move."""
        width, height = self.map[0], self.map[1]
        size = width * height

        while True:
            # Pick a random tile
            start_index = randint(0, size - 1)

            # If we own the tile, make a random move starting from it
            if self.terrain[start_index] == self.player_index:
                row = start_index // width
                col = start_index % width
                end_index = start_index

                rand = random()
                if rand < 0.25 and col > 0:
                    end_index -= 1  # left
                elif rand < 0.5 and col < width - 1:
                    end_index += 1  # right
                elif rand < 0.75 and row < height - 1:
                    end_index += width  # down
                elif row > 0:
                    end_index -= width
                else:
                    continue

                return np.array([start_index, end_index])

    def apply_action(self, action, random=False):
        """Actions are formatted as (from, to) arrays of shape (2,).
        If the action is invalid, a random one will be selected.

        Returns an (observation, reward, done) tuple.

        TODO: potential representations
        (1) (int, int) from/to tuple (when applying, can round to nearest)
        (2) (900 + 900,) array of 0s and 1s (when applying, can use argmax for each of from/to ranges

        TODO: incorporate 50% moves
        """
        if not self._valid(action):
            print('invalid action, choosing one at random')
            action = self.get_random_action()

        start_index, end_index = int(round(action[0])), int(round(action[1]))
        self.attack(start_index, end_index)

        # Generate return info
        obs = self.prev_observation
        next_obs = self.wait_for_next_observation()
        reward = self.reward_func(obs, action, next_obs, self)
        done = next_obs['result'] is not None

        self.reward_history.append(reward)
        return next_obs, reward, done
