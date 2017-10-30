#!/usr/bin/env python

"""
generals.py

Interface for generals.io. Borrows from https://github.com/toshima/generalsio.
"""

import json
import time
import threading
import numpy as np
from websocket import create_connection, WebSocketConnectionClosedException
from rpd_interfaces.interfaces import Interface

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

class Generals(Interface):
    def __init__(self, user_id):
        self.client = Client(SERVER_ENDPOINT)
        self.user_id = user_id

        # Game data
        self.meta = {}  # playerIndex, replay_id, chat_room, team_chat_room, usernames, teams
        self.player_index = -1
        self.cities = []
        self.map = []
        self.stars = []
        self.active = False
        self.move_id = 1

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

        print('waiting to join a %s game' % mode)

    def leave_game(self):
        """For rage quits and such."""
        self.client.send(['leave_game'])

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
        terrain = self.map[size+2:size+2+size]

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
            'terrain': terrain,  # type of each square (-4, -3, -2, -1, or a player index indicating ownership)
            'turn': turn,
            'scores': scores,  # list of dictionaries containing score information for each player
            'cities': self.cities,
        }

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

    def extract_observation(self, update):
        """Return an observation as a tensor."""
        return np.array(self.map)  # TODO

    def attack(self, start_index, end_index, is_half=False):
        if not self.active:
            raise ValueError('no updates have been received')
        self.client.send(['attack', start_index, end_index, is_half, self.move_id])
        self.move_id += 1
