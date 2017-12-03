#!/usr/bin/env python

import numpy as np
from scipy.spatial.distance import cdist
try:
    import queue
except ImportError:
    import multiprocessing as queue

MAP_SIZE = 18

_CITY_MAX_ARMY = 40
_NEUTRAL = 0
_TURN_LIMIT = 100

def land_dt(player, state, next_state, opponent_land_count):
    if state is None:
        return np.count_nonzero(next_state['friendly'])
    return np.count_nonzero(next_state['friendly']) - np.count_nonzero(state['friendly'])

def fifty_squares_acquired(player, state, next_state, opponent_land_count):
    if state is None:
        return 1 if np.count_nonzero(next_state['friendly']) >= 50 else 0
    return 1 if np.count_nonzero(next_state['friendly']) >= 50 else 0

def move_made(player, state, next_state, opponent_land_count):
    """Assumes that the opponent is NOT making moves."""
    if state is None:
        return 0.0
    return int(state['last_location'] != next_state['last_location'])

def win_loss(player, state, next_state, opponent_land_count):
    if not opponent_land_count:
        return 1
    else:
        return 0

class Player(object):
    def __init__(self, id_no, general_loc, reward_fn_name='land_dt'):
        self.id_no = id_no
        self.actions = queue.Queue()
        self.outputs = queue.Queue()
        self.rewards = queue.Queue()
        self.last_state = None
        self.last_location = general_loc
        self.reward_fn = eval(reward_fn_name)
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
    def __init__(self, reward_fn_name=None):
        self.width = MAP_SIZE
        self.height = MAP_SIZE
        self.reward_fn_name = reward_fn_name

        self.mountains = np.zeros((MAP_SIZE, MAP_SIZE))
        self.cities = np.zeros((MAP_SIZE, MAP_SIZE))
        self.generals = np.zeros((MAP_SIZE, MAP_SIZE))
        self.armies = np.zeros((MAP_SIZE, MAP_SIZE))
        self.owner = np.zeros((MAP_SIZE, MAP_SIZE))

        self.turn_count = 1
        self.grid = np.vstack(np.mgrid[:self.width, :self.height].T)
        self.num_players = 0
        self.players = {}
        self.cities_list = []
        self.generals_list = []

    def add_army(self, pos, player, army):
        self.owner[pos[0], pos[1]] = player
        self.armies[pos[0], pos[1]] = army

    def add_mountain(self, pos):
        self.mountains[pos[0], pos[1]] = 1

    def add_city(self, pos, army):
        self.cities[pos[0], pos[1]] = 1
        self.armies[pos[0], pos[1]] = army
        self.cities_list.append(pos)

    def add_general(self, pos, player_id=-1):
        self.num_players += 1
        if player_id == -1:
            player_id = self.num_players
        self.generals[pos[0], pos[1]] = 1
        self.owner[pos[0], pos[1]] = player_id
        self.armies[pos[0], pos[1]] = 1
        if self.reward_fn_name is not None:
            self.players[player_id] = Player(player_id, pos, self.reward_fn_name)
        else:
            self.players[player_id] = Player(player_id, pos)
        self.generals_list.append(pos)

    def update(self, fast_mode=False):
        for player in self.players.values():
            if not player.actions.empty():
                start_location, end_location = player.get_action()
                self._execute_action(player, start_location, end_location)
        self._spawn()
        if not fast_mode: # ?
            for player in self.players.values():
                self._generate_obs(player)
        self.turn_count += 1

    def is_valid_action(self, player, start_location, end_location):
        """Does not do boundary checks."""
        s_x, s_y = start_location
        e_x, e_y = end_location
        return self.owner[s_x, s_y] == player.id_no \
               and self.armies[s_x, s_y] > 1 and np.abs(s_x - e_x) + np.abs(s_y - e_y) == 1

    def _execute_action(self, player, start_location, end_location):
        if start_location == end_location:
            return
        s_x, s_y = start_location
        e_x, e_y = end_location  # and self.terrain[e_x, e_y] != _MOUNTAIN
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
                    if self.generals[e_x, e_y]:
                        defeated = self.owner[e_x, e_y]
                        defeated_map = (self.owner == defeated)
                        diff = defeated - player.id_no
                        self.owner -= defeated_map * diff
                        self.num_players -= 1
                    self.armies[e_x, e_y] *= -1
                    prev_owner = self.owner[e_x, e_y]
                    self.owner[e_x, e_y] = player.id_no
            player.update_location(end_location)

            # if self.mountains[e_x, e_y]:
            #     player.invalid_penalty = -1
        else:
            pass
            # player.invalid_penalty = -1
            # print("Invalid action {} -> {}".format(start_location, end_location))

    def _generate_obs(self, player):
        friendly = self.owner == player.id_no
        neutral = self.owner == _NEUTRAL
        enemy = np.logical_xor(np.logical_not(friendly), neutral)
        player_owned = np.transpose(friendly.nonzero())
        distances = np.min(cdist(self.grid, player_owned, 'euclidean'), axis=1).reshape(self.height, self.width).T
        seen = (distances <= 1.5).astype(np.uint8)
        
        visible_mountains = seen * self.mountains
        visible_generals = seen * self.generals
        fog = 1 - seen
        hidden_terrain = fog * (self.cities + self.mountains)
        visible_armies = seen * self.armies
        visible_friendly = visible_armies * friendly
        visible_enemy = visible_armies * enemy
        visible_cities = visible_armies * self.cities

        opponent_land_count = np.sum(enemy)
        opponent_army_count = np.sum(self.armies * enemy)

        new_state = {
            'mountains':      visible_mountains,
            'generals':       visible_generals,
            'hidden_terrain': hidden_terrain,
            'fog':            fog,
            'friendly':       visible_friendly,
            'enemy':          visible_enemy,
            'cities':         visible_cities,
            'opp_land':       opponent_land_count,
            'opp_army':       opponent_army_count,
            'last_location':  player.last_location
        }
        reward = player.reward_fn(player, player.last_state, new_state, opponent_land_count) + player.invalid_penalty
        done = self.owner[player.general_loc] != player.id_no or self.num_players == 1 or self.turn_count >= _TURN_LIMIT
        player.set_output((new_state, reward, done))

    def _spawn(self):
        for x, y in self.cities_list:
            if self.owner[x, y] != _NEUTRAL or self.armies[x, y] < _CITY_MAX_ARMY:
                self.armies[x, y] += 1
        for x, y in self.generals_list:
            self.armies[x, y] += 1

        if self.turn_count % 25 == 0:
            player_owned = self.owner > 0
            self.armies += player_owned

    def action(self, player_id, start_location, end_location):  # is_half, player_id
        if player_id in self.players:
            if 0 <= end_location[0] < MAP_SIZE and 0 <= end_location[1] < MAP_SIZE:
                self.players[player_id].set_action((start_location, end_location))
            return
        raise ValueError

    def action_simple(self, player_id, direction):
        if player_id in self.players:
            start_location = self.players[player_id].last_location
            if self.owner[start_location] != player_id or self.armies[start_location] <= 2:
                start_location = self.players[player_id].general_loc
            end_location = (start_location[0] + direction[0], start_location[1] + direction[1])
            if 0 <= end_location[0] < MAP_SIZE and 0 <= end_location[1] < MAP_SIZE:
                self.players[player_id].set_action((start_location, end_location))
            return
        raise ValueError
