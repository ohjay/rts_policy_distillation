#!/usr/bin/env python

"""
generals_sim.py

Interface for the simulated version of generals.io.
This version is local and doesn't require any networking.
"""

from random import randint
import json
import os
from PIL import Image, ImageDraw, ImageFont

from rpd_interfaces.generals.simulator import *
from rpd_interfaces.interfaces import Environment


# Preprocessors
# -------------
def add_dimension(obs):
    if obs is not None:
        for _name in ('mountains', 'generals', 'hidden_terrain', 'fog',
                      'friendly', 'enemy', 'cities', 'opp_land', 'opp_army'):
            obs[_name] = obs[_name][..., None]
    return obs


class GeneralsEnv(Environment):
    def __init__(self, root_dir, reward_fn_name=None):
        listdir = os.listdir(root_dir)
        self.replays = [root_dir + file for file in listdir if file.endswith('.gioreplay')]
        self.map = None
        self.reward_fn_name = reward_fn_name

    def reset(self, map_init='random', player_id=1, preprocessors=()):
        """Sets the map using a random replay"""
        if map_init.lower() == 'empty':
            self.map = self._get_random_map(include_mountains=False, include_cities=False)
        elif map_init.lower() == 'random':
            self.map = self._get_random_map()
        else:
            raise NotImplementedError('map init "%s" not supported' % map_init)
        self.map.update()
        out = self.map.players[player_id].get_output()
        for prep in preprocessors:
            out = (eval(prep)(out[0]),) + out[1:]
        return out

    def step(self, action1, action2=None, player_id=1, preprocessors=()):
        """Takes two tuples of (x, y, dir).
        Action 1 is for player 1, 2 is for player 2
        Returns: (state1, reward1, dead1), (state2, reward2, dead2)
        """
        start_loc1 = (action1[0], action1[1])
        self.map.action(1, start_loc1, self._get_movement_for_action(start_loc1, action1[2]))
        if action2:
            start_loc2 = (action2[0], action2[1])
            self.map.action(2, start_loc2, self._get_movement_for_action(start_loc2, action2[2]))
        self.map.update()
        out1 = self.map.players[1].get_output()
        out2 = None
        if action2:
            out2 = self.map.players[2].get_output()
        for prep in preprocessors:
            _prep = eval(prep)
            out1 = (_prep(out1[0]),) + out1[1:]
            if out2 is not None:
                out2 = (_prep(out2[0]),) + out2[1:]
        return out1 if player_id == 1 else out2

    def step_simple(self, action1, action2=None, player_id=1, preprocessors=()):
        """Takes two directions, and moves each agent in the corresponding direction."""
        self.map.action_simple(1, self._get_movement_for_action((0, 0), action1))
        if action2:
            self.map.action_simple(2, self._get_movement_for_action((0, 0), action2))
        self.map.update()
        out1 = self.map.players[1].get_output()
        out2 = None
        if action2:
            out2 = self.map.players[2].get_output()
        for prep in preprocessors:
            _prep = eval(prep)
            out1 = (_prep(out1[0]),) + out1[1:]
            if out2 is not None:
                out2 = (_prep(out2[0]),) + out2[1:]
        return out1 if player_id == 1 else out2

    def _get_movement_for_action(self, initial, direction):
        """Maps 0, 1, 2, 3, 4 to up, down, left, right, stay"""
        dirs = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}
        return initial[0] + dirs[direction][0], initial[1] + dirs[direction][1]

    def _get_random_map(self, include_mountains=True, include_cities=True):
        while True:
            index = randint(0, len(self.replays) - 1)
            replay = json.load(open(self.replays[index]))
            if len(replay['usernames']) == 2 and replay['mapWidth'] == 18 and replay['mapHeight'] == 18:
                break
        m = Map(self.reward_fn_name)
        if include_mountains:
            for mountain in replay['mountains']:
                m.add_mountain(self._flat_to_2d(mountain))
        if include_cities:
            for i in range(len(replay['cities'])):
                m.add_city(self._flat_to_2d(replay['cities'][i]), replay['cityArmies'][i])
        for general in replay['generals']:
            m.add_general(self._flat_to_2d(general))
        return m

    def get_random_action(self, **kwargs):
        """Get a random move.
        If a "semi valid" move is desired, call `get_random_action` as
        `get_random_action(semi_valid=True, player_id=X)`.
        """
        if kwargs.get('semi_valid', False) and 'player_id' in kwargs:
            return self.get_random_semi_valid_action(**{'player_id': kwargs['player_id']})
        return np.array((randint(0, 17), randint(0, 17), randint(0, 3)))

    def get_random_semi_valid_action(self, player_id):
        valid_start = np.logical_and(self.map.owner == player_id, self.map.armies > 1)
        valid_start = np.transpose(valid_start.nonzero())
        start = valid_start[randint(0, len(valid_start) - 1)]
        return np.array((start[0], start[1], randint(0, 3)))

    def is_friendly_square(self, x, y, player_id=1):
        return self.map.owner[x, y] == player_id

    def is_valid_action(self, action):
        start_location = (action[0], action[1])
        end_location = self._get_movement_for_action(start_location, action[2])
        return self.map.is_valid_action(self.map.players[1], start_location, end_location)

    def get_nearest_valid_action(self, action, player_id=1):
        """Returns the action closest to ACTION that is actually valid.
        Only the square will ever be changed: it will become the nearest square that is owned by player PLAYER_ID
        and has > 1 army unit on it. If player PLAYER_ID has no valid moves, the input action will be returned.
        """
        if self.is_valid_action(action):
            return action
        valid_start = np.logical_and(self.map.owner == player_id, self.map.armies > 1)
        valid_start = np.transpose(valid_start.nonzero())
        if len(valid_start) == 0:
            return action
        dist = (valid_start[:, 0] - action[0]) ** 2 + (valid_start[:, 1] - action[1]) ** 2
        start_location = valid_start[dist.argmin()]
        return start_location[0], start_location[1], action[2]

    def get_action_from_q_values(self, q_values, ensure_valid=False):
        if ensure_valid:
            return self.get_valid_action_from_q_values(q_values)
        if len(q_values) == 1:
            q_values = np.reshape(q_values, (MAP_SIZE, MAP_SIZE, 4))
            act_x, act_y, act_dir = np.unravel_index(np.argmax(q_values), q_values.shape)
        elif len(q_values) == 3:
            act_x, act_y = np.argmax(q_values[0]), np.argmax(q_values[1])
            act_dir = np.argmax(q_values[2])
        else:
            raise NotImplementedError('action space not supported')
        return np.array((act_x, act_y, act_dir))

    def get_valid_action_from_q_values(self, q_values, player_id=1):
        """
        If there is one Q-value - (x, y, dir):
        - Select the VALID (x, y, dir) with the highest Q value.

        If there are three Q values - x, y, and dir:
        - Select the VALID (x, y) with the highest q_x[x] + q_y[y].
        - Select the dir with the highest q_dir[dir].
        * assumes x is vertical (first index into NumPy arrays), and y is horizontal
        * assumes Q_X and Q_Y are both (1, 18) arrays
        """
        if len(q_values) == 1:
            q_values = np.reshape(q_values, (MAP_SIZE, MAP_SIZE, 4))
            valid_start = np.logical_and(self.map.owner == player_id, self.map.armies > 1)
            q_values_valid = q_values * np.repeat(valid_start[:, :, None], 4, axis=2)
            act_x, act_y, act_dir = np.unravel_index(np.argmax(q_values_valid), q_values_valid.shape)
        elif len(q_values) == 3:
            q_x, q_y, q_dir = q_values
            q_xy = np.repeat(q_x.T, MAP_SIZE, axis=1) + np.repeat(q_y, MAP_SIZE, axis=0)
            q_xy_valid = q_xy * np.logical_and(self.map.owner == player_id, self.map.armies > 1)
            act_x, act_y = np.unravel_index(np.argmax(q_xy_valid), q_xy_valid.shape)
            act_dir = np.argmax(q_dir)
        else:
            raise NotImplementedError('action space not supported')
        return np.array((act_x, act_y, act_dir))

    def _flat_to_2d(self, index):
        return index // MAP_SIZE, index % MAP_SIZE

    def get_image_of_state(self, state):
        mountains = state['mountains']
        generals = state['generals']
        hidden_terrain = state['hidden_terrain']
        fog = state['fog']
        friendly = state['friendly']
        enemy = state['enemy']
        cities = state['cities']
        last_location = state['last_location']

        size = 400
        step = size / MAP_SIZE
        image = Image.new('RGBA', (size, size), (255, 255, 255, 255))
        d = ImageDraw.Draw(image)
        colors = [
            (255, 255, 255, 255),   # 0 - Empty - white
            (),                     # 1 - General - no color
            (0, 136, 34, 255),      # 2 - City - Green
            (128, 64, 0, 255),      # 3 - Mountain - Brown
            (32, 32, 32, 255),      # 4 - Fog - Dark Grey
            (124, 124, 124, 255),   # 5 - Terrain Fog - Grey
            (0, 0, 255, 255),       # 6 - Blue player
            (255, 0, 0, 255),       # 7 - Red player
            (0, 0, 100, 255)        # 8 - Blue player last location - Dark Blue
        ]
        dir = os.path.dirname(__file__)
        try:
            font = ImageFont.truetype(os.path.join(dir, 'FreeMono.ttf'), 40)
        except IOError as e:
            print('IOError: failed to open font file at %s.' % os.path.join(dir, 'FreeMono.ttf'))
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[0])
                if mountains[i,j]: # Hack
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[3])
                if friendly[i, j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[6])
                    general = ''
                    if generals[i, j]:
                        general = '*'
                    d.text((i * step + step//4, j * step + step//4), str(int(friendly[i, j])) + general, fill=colors[0])
                if enemy[i,j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[7])
                    general = ''
                    if generals[i, j]:
                        general = '*'
                    d.text((i * step + step//4, j * step + step//4), str(int(enemy[i, j])) + general, fill=colors[0])
                if cities[i,j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[2])
                    d.text((i * step + step//4, j * step + step//4), str(int(cities[i, j])), fill=colors[0])
                if mountains[i,j]:
                    d.text((i * step + step//4, j * step + step//4), '^^', fill=colors[0])
                if fog[i,j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[4])
                if hidden_terrain[i,j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[5])
                    d.text((i * step + step//4, j * step + step//4), '??', fill=colors[0])
                if (i,j) == last_location:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[8])
                    general = ''
                    if generals[i, j]:
                        general = '*'
                    d.text((i * step + step//4, j * step + step//4), str(int(friendly[i, j])) + general, fill=colors[0])
        del d
        return image
