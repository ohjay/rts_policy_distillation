#!/usr/bin/env python

"""
generals_sim.py

Interface for the simulated version of generals.io.
This version is local and doesn't require any networking.
"""

from random import randint
import json
import os
import copy
import collections
from PIL import Image, ImageDraw, ImageFont

from rpd_interfaces.generals.simulator import *
from rpd_interfaces.interfaces import Environment

DEFAULT_ACTION_SPACE = [('target_x', [18]), ('target_y', [18]), ('direction', [4])]
DEFAULT_ACTION_SPACE = collections.OrderedDict(DEFAULT_ACTION_SPACE)  # {output_name: shape}
NUM_DIRECTIONS = 4


# Preprocessors
# -------------
def add_dimension(obs):
    if obs is not None:
        for _name in ('mountains', 'generals', 'hidden_terrain', 'fog',
                      'friendly', 'enemy', 'cities'):
            obs[_name] = obs[_name][..., None]
    return obs


class GeneralsEnv(Environment):
    def __init__(self, root_dir, action_space=None):
        self.replays = [root_dir + filename for filename in os.listdir(root_dir) if filename.endswith('.gioreplay')]
        self.map = None
        if type(action_space) == dict:
            self.action_space = collections.OrderedDict()
            for output_name in sorted(action_space.keys(), key=lambda x: action_space[x].get('order', float('inf'))):
                self.action_space[output_name] = action_space[output_name]['shape']
        else:
            self.action_space = DEFAULT_ACTION_SPACE
        print('Loaded action space as %r.' % self.action_space)

    def reset(self, reward_fn_names=(), reward_weights=(),
              map_init='random', player_id=1, preprocessors=(), s_index=None):
        """Sets the map using a random replay."""
        if map_init.lower() == 'empty':
            self.map = self._get_random_map(include_mountains=False, include_cities=False, s_index=s_index,
                                            reward_fn_names=reward_fn_names, reward_weights=reward_weights)
        elif map_init.lower() == 'random':
            self.map = self._get_random_map(s_index=s_index,
                                            reward_fn_names=reward_fn_names, reward_weights=reward_weights)
        else:
            raise NotImplementedError('map init "%s" not supported' % map_init)
        self.map.update()
        out = self.map.players[player_id].get_output()
        for prep in preprocessors:
            out = (eval(prep)(copy.deepcopy(out[0])),) + out[1:]
        return out

    def _encode_action(self, target_x, target_y, direction):
        """Converts (target_x, target_y, direction) into a single action value."""
        if len(self.action_space) == 1:
            # Assume (x, y, dir) all wrapped up in one value
            action = np.array([NUM_DIRECTIONS * (MAP_SIZE * target_x + target_y) + direction])
        elif len(self.action_space) == 2:
            # Assume the two outputs are an xy-point and a direction
            action = np.array([MAP_SIZE * target_x + target_y, direction])
        elif len(self.action_space) == 3:
            # Assume the outputs are all separate
            action = np.array([target_x, target_y, direction])
        else:
            raise NotImplementedError('action space not supported')
        return action

    def _decode_action(self, action):
        """Breaks an action into its (target_x, target_y, direction) components."""
        if len(self.action_space) == 1:
            # Assume (x, y, dir) all wrapped up in one value
            xy_direction = action[0]
            direction = xy_direction % NUM_DIRECTIONS
            xy = xy_direction // 4
            target_x, target_y = xy // MAP_SIZE, xy % MAP_SIZE
        elif len(self.action_space) == 2:
            # Assume the two outputs are an xy-point and a direction
            xy, direction = action[0], action[1]
            target_x, target_y = xy // MAP_SIZE, xy % MAP_SIZE
        elif len(self.action_space) == 3:
            # Assume the outputs are all separate
            target_x, target_y, direction = action[0], action[1], action[2]
        else:
            raise NotImplementedError('action space not supported')
        return target_x, target_y, direction

    def _get_movement_for_action(self, initial, direction):
        """Maps (0, 1, 2, 3, 4) to (up, down, left, right, stay)."""
        dirs = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}
        return initial[0] + dirs[direction][0], initial[1] + dirs[direction][1]

    def _flat_to_2d(self, index):
        return index // MAP_SIZE, index % MAP_SIZE

    def _get_random_map(self, include_mountains=True, include_cities=True,
                        s_index=None, reward_fn_names=(), reward_weights=()):
        while True:
            if s_index is None:
                index = randint(0, len(self.replays) - 1)
            else:
                index = s_index
                s_index += 1
            replay = json.load(open(self.replays[index]))
            if len(replay['usernames']) == 2 and replay['mapWidth'] == 18 and replay['mapHeight'] == 18:
                break
        m = Map()
        if include_mountains:
            for mountain in replay['mountains']:
                m.add_mountain(self._flat_to_2d(mountain))
        if include_cities:
            for i in range(len(replay['cities'])):
                m.add_city(self._flat_to_2d(replay['cities'][i]), replay['cityArmies'][i])
        for general in replay['generals']:
            m.add_general(self._flat_to_2d(general), reward_fn_names=reward_fn_names, reward_weights=reward_weights)
        return m

    def step(self, action1, action2=None, player_id=1, preprocessors=()):
        """Takes two tuples of (x, y, dir).
        Action 1 is for player 1, 2 is for player 2
        Returns: (state1, reward1, dead1), (state2, reward2, dead2)
        """
        target_x1, target_y1, direction1 = self._decode_action(action1)
        start_loc1 = (target_x1, target_y1)
        self.map.action(1, start_loc1, self._get_movement_for_action(start_loc1, direction1))
        if action2 is not None:
            target_x2, target_y2, direction2 = self._decode_action(action2)
            start_loc2 = (target_x2, target_y2)
            self.map.action(2, start_loc2, self._get_movement_for_action(start_loc2, direction2))
        self.map.update()
        out1 = self.map.players[1].get_output()
        out2 = None
        if action2 is not None:
            out2 = self.map.players[2].get_output()
        for prep in preprocessors:
            _prep = eval(prep)
            out1 = (_prep(copy.deepcopy(out1[0])),) + out1[1:]
            if out2 is not None:
                out2 = (_prep(copy.deepcopy(out2[0])),) + out2[1:]
        return out1 if player_id == 1 else out2

    def get_random_action(self, **kwargs):
        """Get a random move.
        If a "semi valid" move is desired, call `get_random_action` as
        `get_random_action(semi_valid=True, player_id=X)`.
        """
        if kwargs.get('semi_valid', False) and 'player_id' in kwargs:
            return self.get_random_semi_valid_action(**{'player_id': kwargs['player_id']})
        return np.array([randint(0, shape[0] - 1) for shape in self.action_space.values()])

    def get_random_semi_valid_action(self, player_id):
        valid_start = np.logical_and(self.map.owner == player_id, self.map.armies > 1)
        valid_start = np.transpose(valid_start.nonzero())
        start = valid_start[randint(0, len(valid_start) - 1)]
        random_dir = randint(0, 3)
        return self._encode_action(start[0], start[1], random_dir)

    def is_valid_action(self, action):
        target_x, target_y, direction = self._decode_action(action)
        start_location = (target_x, target_y)
        end_location = self._get_movement_for_action(start_location, direction)
        return self.map.is_valid_action(self.map.players[1], start_location, end_location)

    def get_action_from_q_values(self, q_values, ensure_valid=False):
        if ensure_valid:
            return self.get_valid_action_from_q_values(q_values)

        # Q_VALUES is assumed to be a list containing the set of Q values for each output
        return np.array([np.argmax(q_values[i].flatten()) for i in range(len(self.action_space))])

    def get_valid_action_from_q_values(self, q_values, player_id=1):
        valid_start = np.logical_and(self.map.owner == player_id, self.map.armies > 1)
        valid_start_1d = valid_start.flatten('F')  # column-major because we assume that x is horizontal
        if len(self.action_space) == 1:
            q_values_valid = [q_values[0].flatten() * np.repeat(valid_start_1d, NUM_DIRECTIONS)]
            return np.array([np.argmax(q_values_valid[0].flatten())])
        elif len(self.action_space) == 2:
            q_xy_valid = q_values[0].flatten() * valid_start_1d
            return np.array([np.argmax(q_xy_valid), np.argmax(q_values[1].flatten())])
        elif len(self.action_space) == 3:
            q_x, q_y, q_dir = q_values
            q_x, q_y = q_x.flatten(), q_y.flatten()
            q_xy = np.repeat(q_x[None, :], MAP_SIZE, axis=0) + np.repeat(q_y[:, None], MAP_SIZE, axis=1)
            q_xy_valid = (q_xy * valid_start).flatten('F')
            target_xy = np.argmax(q_xy_valid)
            target_x, target_y = target_xy // MAP_SIZE, target_xy % MAP_SIZE
            return np.array([target_x, target_y, np.argmax(q_dir)])
        else:
            raise NotImplementedError('action space not supported')

    def get_image_of_state(self, state):
        mountains = state['mountains']
        generals = state['generals']
        hidden_terrain = state['hidden_terrain']
        fog = state['fog']
        friendly = state['friendly']
        enemy = state['enemy']
        cities = state['cities']
        last_location = state['last_location']

        px_size = 400
        step = px_size / MAP_SIZE
        image = Image.new('RGBA', (px_size, px_size), (255, 255, 255, 255))
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
        font_dir = os.path.dirname(__file__)
        try:
            font = ImageFont.truetype(os.path.join(font_dir, 'FreeMono.ttf'), 40)
        except IOError as e:
            print('IOError: failed to open font file at %s.' % os.path.join(font_dir, 'FreeMono.ttf'))
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[0])
                if mountains[i, j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[3])
                if cities[i, j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[2])
                    d.text((i * step + step // 4, j * step + step // 4), str(int(cities[i, j])), fill=colors[0])
                if friendly[i, j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[6])
                    general = ''
                    if generals[i, j]:
                        general = '*'
                    d.text((i * step + step // 4, j * step + step // 4), str(int(friendly[i, j])) + general, fill=colors[0])
                if enemy[i, j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[7])
                    general = ''
                    if generals[i, j]:
                        general = '*'
                    d.text((i * step + step // 4, j * step + step // 4), str(int(enemy[i, j])) + general, fill=colors[0])
                if mountains[i, j]:
                    d.text((i * step + step // 4, j * step + step // 4), '^^', fill=colors[0])
                if fog[i, j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[4])
                if hidden_terrain[i, j]:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[5])
                    d.text((i * step + step // 4, j * step + step // 4), '??', fill=colors[0])
                if (i, j) == last_location:
                    d.rectangle((i * step, j * step, i * step + step, j * step + step), fill=colors[8])
                    general = ''
                    if generals[i, j]:
                        general = '*'
                    d.text((i * step + step // 4, j * step + step // 4), str(int(friendly[i, j])) + general, fill=colors[0])
        del d
        return image
