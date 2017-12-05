#!/usr/bin/env python

"""
generals/reward.py

Reward functions for generals.io.

The reward function interface differs between real and sim:

- real
  A reward function R(s, a, s') should accept (observation, action, resulting observation, environment) as input,
  and return a scalar value representing the "goodness" of the given situation.

- sim
  A reward function R(p, s, a, s', olc) should accept (player, obs, action, resulting obs, opponent land ct.) as input,
  and return a scalar value representing the "goodness" of the given situation for the player.
"""

import numpy as np


# REAL REWARD FUNCTIONS
# ---------------------

def rew_total_units(obs, action, next_obs, env):
    """Assigns reward based on the total number of units owned.
    Questionable since this goes up with time regardless... (?)."""
    return next_obs['other'][11 + env.player_index * 3]


def rew_total_land_dt(obs, action, next_obs, env):
    """Assigns reward based on the amount of land acquired by taking the given action."""
    _tiles_idx = 10 + env.player_index * 3
    return next_obs['other'][_tiles_idx] - obs['other'][_tiles_idx]


REAL_REWARD_FUNCTIONS = {
    'rew_total_units':   rew_total_units,
    'rew_total_land_dt': rew_total_land_dt,
}


# SIM REWARD FUNCTIONS
# --------------------

def land_dt(player, state, action, next_state, opponent_land_count):
    if state is None:
        return np.count_nonzero(next_state['friendly'])
    return np.count_nonzero(next_state['friendly']) - np.count_nonzero(state['friendly'])


def x_squares_acquired(player, state, action, next_state, opponent_land_count):
    """Goal: acquire X squares as quickly as possible."""
    if state is None:
        return 1 if np.count_nonzero(next_state['friendly']) >= 20 else 0
    return 1 if np.count_nonzero(next_state['friendly']) >= 20 else 0


def move_made(player, state, action, next_state, opponent_land_count):
    """Assumes that the opponent is NOT making moves."""
    if state is None:
        return 0.0
    return int(state['last_location'] != next_state['last_location'])


def win_loss(player, state, action, next_state, opponent_land_count):
    return int(not opponent_land_count)


def best_point(player, state, action, next_state, opponent_land_count):
    if action:
        return player.last_move / np.max(state['friendly'])
    return 0


def vision_gain(player, state, action, next_state, opponent_land_count):
    if state is None:
        fog = next_state['fog']  # type: np.ndarray
        return np.sum(1 - fog) / 3
    return (np.sum(state['fog']) - np.sum(next_state['fog'])) / 3


def army_size(player, state, action, next_state, opponent_land_count):
    if state is None:
        return np.sum(next_state['friendly'])
    return np.sum(next_state['friendly']) - np.count_nonzero(state['friendly'])


SIM_REWARD_FUNCTIONS = {
    'land_dt':            land_dt,
    'x_squares_acquired': x_squares_acquired,
    'move_made':          move_made,
    'win_loss':           win_loss,
    'best_point':         best_point,
    'vision_gain':        vision_gain,
    'army_size':          army_size,
}
