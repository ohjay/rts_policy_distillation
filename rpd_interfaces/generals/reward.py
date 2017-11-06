#!/usr/bin/env python

"""
generals/reward.py

Reward functions for generals.io.

A reward function R(s, a, s') should accept (observation, action, resulting observation, environment) as input,
and return a scalar value representing the "goodness" of the given situation.
"""

def rew_total_units(obs, action, next_obs, env):
    """Assigns reward based on the total number of units owned.
    Questionable since this goes up with time regardless... (?)."""
    return next_obs[11 + env.player_index * 3]

def rew_total_land(obs, action, next_obs, env):
    """Assigns reward based on total amount of land owned."""
    return next_obs[10 + env.player_index * 3]

def rew_total_land_dt(obs, action, next_obs, env):
    """Assigns reward based on the amount of land acquired by taking the given action."""
    _tiles_idx = 10 + env.player_index * 3
    return next_obs[_tiles_idx] - obs[_tiles_idx]
