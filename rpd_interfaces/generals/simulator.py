#!/usr/bin/env python

import numpy as np
from scipy.spatial.distance import cdist
import queue

# Terrain
_EMPTY = 0
_GENERAL = 1
_CITY = 2
_MOUNTAIN = 3
_FOG = 4
_FOGGY_CITY = 6
_FOGGY_MOUNTAIN = 7
_NEUTRAL = 0

_MIN_CITY_DIST = 5
_MIN_GENERAL_DIST = 10

_CITY_MAX_ARMY = 40


def land_dt(player, state, next_state):
    if state is None:
        return np.sum(next_state[1] == player.id_no)
    return np.sum(next_state[1] == player.id_no) - np.sum(state[1] == player.id_no)

class Player(object):
    def __init__(self, id_no, general_loc, reward_fn=land_dt):
        self.id_no = id_no
        self.actions = queue.Queue()
        self.outputs = queue.Queue()
        self.last_state = None
        self.reward_fn = reward_fn
        self.general_loc = general_loc.tolist()

    def set_action(self, action):
        self.actions.put(action)

    def get_action(self):
        return self.actions.get()

    def set_output(self, obs):
        self.outputs.put(obs)

    def get_output(self):
        new_state = self.outputs.get()
        self.last_state = new_state
        return new_state

class Map(object):
    def __init__(self, size, player_count):
        self.width = size[0]
        self.height = size[1]

        self.terrain = np.zeros(size)
        self.armies = np.zeros(size)
        self.owner = np.zeros(size)
        self.turn_count = 1
        self.grid = np.vstack(np.mgrid[:self.width, :self.height].T)

        for _ in range(self.width * self.height // 3):
            self.terrain[np.random.randint(self.width), np.random.randint(self.height)] = _MOUNTAIN

        self.cities = np.hstack((np.random.randint(self.width, size = (2 * player_count, 1)), \
                                np.random.randint(self.height, size = (2 * player_count, 1))))

        for x,y in self.cities:
            self.terrain[x,y] = _CITY
            self.armies[x,y] = _CITY_MAX_ARMY

        self.generals = np.hstack((np.random.randint(self.width, size = (player_count, 1)), \
                                np.random.randint(self.height, size = (player_count, 1))))

        for i, loc in enumerate(self.generals):
            x, y = loc
            self.terrain[x,y] = _GENERAL
            self.owner[x,y] = i + 1

        self.players = {i: Player(i, self.generals[i-1]) for i in range(1, player_count + 1)}
        self.remaining_players = player_count
        """
        self.cities = np.zeros((player_count * 2, 2))
        self.generals = np.zeros((player_count, 2))
        for i in range(player_count):
            self.cities[2*i] = np.array([np.random.randint(self.width), np.random.randint(self.height)])
        """

    def _update(self):
        for player in self.players.values():
            start_location, end_location = player.get_action()
            self._execute_action(player, start_location, end_location)
        for player in self.players.values():
            self._generate_obs(player)
        self._spawn()
        self.turn_count += 1
        self.print_state()
        if self.remaining_players > 1:
            self._update()

    def _execute_action(self, player, start_location, end_location):
        s_x, s_y = start_location
        e_x, e_y = end_location
        if self.owner[s_x, s_y] == player.id_no \
            and self.armies[s_x, s_y] > 1 \
            and np.abs(s_x - e_x) + np.abs(s_y - e_y) == 1: 
            moving = self.armies[s_x, s_y] - 1
            self.armies[s_x, s_y] = 1
            if self.owner[e_x, e_y] == player.id_no:
                self.armies[e_x, e_y] += moving
            else:
                self.armies[e_x, e_y] -= moving
                if self.armies[e_x, e_y] < 0: # If target square has value 0, does it go neutral?
                    if self.terrain[e_x, e_y] == _GENERAL:
                        defeated = self.owner[e_x, e_y]
                        defeated_map = (self.owner == defeated)
                        diff = defeated - player.id_no
                        self.owner -= defeated_map * diff
                        self.remaining_players -= 1
                    self.armies[e_x, e_y] *= -1
                    self.owner[e_x, e_y] = player.id_no
        else:
            print("Invalid action {} {}".format(start_location, end_location))

    def _generate_obs(self, player):
        player_owned = np.transpose((self.owner == player.id_no).nonzero())
        distances = np.min(cdist(self.grid, player_owned, 'cityblock'), axis=1).reshape(self.height, self.width).T
        seen = distances <= 1
        fog = 1 - seen
        visible_terrain = self.terrain * seen + fog * _FOG + self.terrain * (self.terrain != _GENERAL) * fog
        visible_armies = self.armies * seen
        visible_owner = self.owner * seen
        new_state = (visible_terrain, visible_armies, visible_owner,)
        reward = player.reward_fn(player, player.last_state, new_state)
        done = self.owner[player.general_loc] == player.id_no
        player.set_output((new_state, reward, done))

    def _spawn(self):
        for x,y in self.cities:
            if self.owner[x,y] != _NEUTRAL or self.armies[x,y] < _CITY_MAX_ARMY:
                self.armies[x,y] += 1
        for x,y in self.generals:
            self.armies[x,y] += 1

        if self.turn_count % 25 == 0:
            player_owned = (self.owner > 0)
            self.armies += player_owned

    def action(self, player_id, start_location, end_location): # is_half, player_id
        if player_id in self.players:
            self.players[player_id].set_action((start_location, end_location))
            return self.players[player_id].get_output()
        print("Invalid player %d", player_id)
        raise

    def __str__(self):
        return "{} \n {} \n {} \n".format(self.terrain, self.armies, self.owner)

    def print_state(self):
        print(self.terrain)
        print(self.owner)
        print(self.armies)