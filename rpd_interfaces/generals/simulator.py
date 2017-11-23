#!/usr/bin/env python

import numpy as np
from scipy.spatial.distance import cdist
import queue
from random import randint
import json
import os

import matplotlib
from PIL import Image, ImageDraw, ImageFont
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Terrain
_EMPTY = 0
_GENERAL = 1
_CITY = 2
_MOUNTAIN = 3
_FOG = 4
_FOGGY_TERRAIN = 5
_FOGGY_CITY = 6
_FOGGY_MOUNTAIN = 7
_NEUTRAL = 0

_CITY_MAX_ARMY = 40

_MAP_SIZE = 18

_TURN_LIMIT = 50


def land_dt(player, state, next_state, opponent_land_count):
    if state is None:
        return np.sum(next_state['ownership'] == player.id_no)
    return np.sum(next_state['ownership'] == player.id_no) - np.sum(state['ownership'] == player.id_no)

def win_loss(player, state, next_state, opponent_land_count):
    if not opponent_land_count:
        return 1
    else:
        return 0

class Player(object):
    def __init__(self, id_no, general_loc, reward_fn=land_dt):
        self.id_no = id_no
        self.actions = queue.Queue()
        self.outputs = queue.Queue()
        self.rewards = queue.Queue()
        self.last_state = None
        self.last_location = general_loc
        self.reward_fn = reward_fn
        self.general_loc = general_loc
        self.invalid_penalty = 0

    def set_action(self, action):
        self.actions.put(action)

    def get_action(self):
        return self.actions.get()

    def set_output(self, obs):
        self.outputs.put(obs)

    def get_output(self):
        new_state = self.outputs.get()
        self.last_state = new_state[0]
        self.invalid_penalty = 0
        return new_state

    def update_location(self, location):
        self.last_location = location

class Map(object):
    def __init__(self):
        self.width = _MAP_SIZE
        self.height = _MAP_SIZE

        self.terrain = np.zeros((_MAP_SIZE, _MAP_SIZE))
        self.armies = np.zeros((_MAP_SIZE, _MAP_SIZE))
        self.owner = np.zeros((_MAP_SIZE, _MAP_SIZE))
        self.turn_count = 1
        self.grid = np.vstack(np.mgrid[:self.width, :self.height].T)
        self.num_players = 0
        self.players = {}
        self.cities = []
        self.generals = []

    def add_army(self, pos, player, army):
        self.owner[pos[0], pos[1]] = player
        self.armies[pos[0], pos[1]] = army

    def add_mountain(self, pos):
        self.terrain[pos[0], pos[1]] = _MOUNTAIN

    def add_city(self, pos, army):
        self.terrain[pos[0], pos[1]] = _CITY
        self.armies[pos[0], pos[1]] = army
        self.cities.append(pos)

    def add_general(self, pos, player_id=-1):
        self.num_players += 1
        if player_id == -1:
            player_id = self.num_players
        self.terrain[pos[0], pos[1]] = _GENERAL
        self.owner[pos[0], pos[1]] = player_id
        self.armies[pos[0], pos[1]] = 1
        self.players[player_id] = Player(player_id, pos)
        self.generals.append(pos)

    def update(self, fast_mode=False):
        for player in self.players.values():
            if not player.actions.empty():
                start_location, end_location = player.get_action()
                self._execute_action(player, start_location, end_location)
        self._spawn()
        if not fast_mode:
            for player in self.players.values():
                self._generate_obs(player)
        self.turn_count += 1

    def _execute_action(self, player, start_location, end_location):
        if start_location == end_location:
            return
        s_x, s_y = start_location
        e_x, e_y = end_location # and self.terrain[e_x, e_y] != _MOUNTAIN
        if self.owner[s_x, s_y] == player.id_no \
            and self.armies[s_x, s_y] > 1 \
            and np.abs(s_x - e_x) + np.abs(s_y - e_y) == 1: 
            moving = self.armies[s_x, s_y] - 1
            self.armies[s_x, s_y] = 1
            if self.owner[e_x, e_y] == player.id_no:
                self.armies[e_x, e_y] += moving
            else:
                self.armies[e_x, e_y] -= moving
                if self.armies[e_x, e_y] < 0:
                    if self.terrain[e_x, e_y] == _GENERAL:
                        defeated = self.owner[e_x, e_y]
                        defeated_map = (self.owner == defeated)
                        diff = defeated - player.id_no
                        self.owner -= defeated_map * diff
                        self.num_players -= 1
                    self.armies[e_x, e_y] *= -1
                    prev_owner = self.owner[e_x, e_y]
                    self.owner[e_x, e_y] = player.id_no
            player.update_location(end_location)
            if self.terrain[e_x, e_y] == _MOUNTAIN:
                player.invalid_penalty = -1
        else:
            player.invalid_penalty = -1
            # print("Invalid action {} -> {}".format(start_location, end_location))

    def _generate_obs(self, player):
        done = self.owner[player.general_loc] != player.id_no or self.num_players == 1

        if self.turn_count >= _TURN_LIMIT or done:
            player.set_output(({}, 0, True))
            return

        player_owned = np.transpose((self.owner == player.id_no).nonzero())
        distances = np.min(cdist(self.grid, player_owned, 'euclidean'), axis=1).reshape(self.height, self.width).T
        seen = distances <= 1.5
        fog = 1 - seen
        visible_terrain = self.terrain * seen + fog * _FOG + self.terrain * (self.terrain != _GENERAL) * fog
        visible_terrain[visible_terrain == _FOGGY_CITY] = _FOGGY_TERRAIN
        visible_terrain[visible_terrain == _FOGGY_MOUNTAIN] = _FOGGY_TERRAIN
        visible_armies = self.armies * seen
        visible_owner = self.owner * seen
        opponent_owned = self.owner == self.other_player(player.id_no)
        opponent_land_count = np.sum(opponent_owned)
        opponent_army_count = np.sum(self.armies[opponent_owned])
        new_state = {'terrain': visible_terrain, 'ownership': visible_owner, 'armies': visible_armies, \
                     'opp_land': opponent_land_count, 'opp_army': opponent_army_count, 'last_location': player.last_location}
        reward = player.reward_fn(player, player.last_state, new_state, opponent_land_count) + player.invalid_penalty
        player.set_output((new_state, reward, done))

    def _spawn(self):
        for x,y in self.cities:
            if self.owner[x,y] != _NEUTRAL: #or self.armies[x,y] < _CITY_MAX_ARMY:
                self.armies[x,y] += 1
        for x,y in self.generals:
            self.armies[x,y] += 1

        if self.turn_count % 25 == 0:
            player_owned = (self.owner > 0)
            self.armies += player_owned

    def action(self, player_id, start_location, end_location): # is_half, player_id
        if player_id in self.players and self.owner[start_location] == player_id:
            if 0 <= end_location[0] < _MAP_SIZE and 0 <= end_location[1] < _MAP_SIZE and self.armies[start_location] > 1:
                self.players[player_id].set_action((start_location, end_location))
            return
        raise ValueError

    def action_simple(self, player_id, direction):
        if player_id in self.players:
            start_location = self.players[player_id].last_location
            if self.owner[start_location] != player_id or self.armies[start_location] <= 2:
                start_location = self.players[player_id].general_loc
            end_location = (start_location[0] + direction[0], start_location[1] + direction[1])
            if 0 <= end_location[0] < _MAP_SIZE and 0 <= end_location[1] < _MAP_SIZE and self.armies[start_location] > 1:
                self.players[player_id].set_action((start_location, end_location))
            return
        raise ValueError

    def other_player(self, player_id):
        if player_id == 1:
            return 2
        return 1

    def __str__(self):
        return "{} \n {} \n {} \n".format(self.terrain, self.armies, self.owner)

class GeneralsEnv:
    def __init__(self, root_dir):
        listdir = os.listdir(root_dir)
        self.replays = [root_dir + file for file in listdir if file.endswith('.gioreplay')]

    def reset(self):
        """Sets the map using a random replay"""
        self.map = self._get_random_map()
        self.map.update()
        return [x.get_output() for x in self.map.players.values()]

    def step(self, action1, action2):
        """Takes two tuples of (start_location, end_location).
        Action 1 is for player 1, 2 is for player 2
        Returns: (state1, reward1, dead1), (state2, reward2, dead2)
        """
        self.map.action(1, action1[0], self._get_movement_for_action(action1[0], action1[1]))
        self.map.action(2, action2[0], self._get_movement_for_action(action2[0], action2[1]))
        self.map.update()
        return [x.get_output() for x in self.map.players.values()]

    def step_simple(self, action1, action2):
        """Takes two directions, and moves each agent in the corresponding direction"""
        self.map.action_simple(1, self._get_movement_for_action((0, 0), action1))
        self.map.action_simple(2, self._get_movement_for_action((0, 0), action2))
        self.map.update()
        return [x.get_output() for x in self.map.players.values()]

    def _get_movement_for_action(self, initial, direction):
        """Maps 0, 1, 2, 3, 4 to up, down, left, right, stay"""
        dirs = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}
        return initial[0] + dirs[direction][0], initial[1] + dirs[direction][1]

    def _get_random_map(self):
        while True:
            index = randint(0, len(self.replays) - 1)
            replay = json.load(open(self.replays[index]))
            if len(replay['usernames']) == 2 and replay['mapWidth'] == 18 and replay['mapHeight'] == 18:
                break
        m = Map()
        for mountain in replay['mountains']:
            m.add_mountain(self._flat_to_2d(mountain))
        for i in range(len(replay['cities'])):
            m.add_city(self._flat_to_2d(replay['cities'][i]), replay['cityArmies'][i])
        for general in replay['generals']:
            m.add_general(self._flat_to_2d(general))
        return m

    def get_random_action(self):
        """Get a random move."""
        return randint(0, 4)

    def _flat_to_2d(self, index):
        return index // _MAP_SIZE, index % _MAP_SIZE

    def get_image_of_state(self, state):
        terrain, armies, owner = state['terrain'], state['armies'], state['ownership']
        size = 400
        step = size / _MAP_SIZE
        image = Image.new('RGBA', (size, size), (255, 255, 255, 255))
        d = ImageDraw.Draw(image)
        colors = [
            (255, 255, 255, 255), # Empty - white
            (), # General - no color
            (0, 136, 34, 255), # City - Green
            (128, 64, 0, 255), # Mountain - Brown
            (32, 32, 32, 255), # Fog - Dark Grey
            (124, 124, 124, 255), # Terrain Fog - Grey
            (0, 0, 255, 255), # Blue player
            (255, 0, 0, 255) # Red player
        ]
        dir = os.path.dirname(__file__)
        font = ImageFont.truetype(os.path.join(dir, 'FreeMono.ttf'), 40)
        for i in range(_MAP_SIZE):
            for j in range(_MAP_SIZE):
                if owner[i, j] > 0:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[int(owner[i, j]) + 5])
                else:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[int(terrain[i, j])])
                if armies[i, j] > 0:
                    general = ''
                    if terrain[i, j] == _GENERAL:
                        general = '*'
                    d.text((i * step + step//4, j * step + step//4), str(int(armies[i, j])) + general, fill=colors[0])
                if terrain[i, j] == _MOUNTAIN:
                    d.text((i * step + step//4, j * step + step//4), '^^', fill=colors[0])
                if terrain[i, j] == _FOGGY_TERRAIN:
                    d.text((i * step + step//4, j * step + step//4), '??', fill=colors[0])
        del d
        return image