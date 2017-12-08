#!/usr/bin/env python

"""
distillation.py

Offline policy distillation implementation.
"""

# TODO
# run DQN training ? with special params ? and collect dataset of (state, q_values) pairs from DQN
# such will represent the teacher
# then construct a student model / optimize with loss as KL divergence b/e teacher & student dists
# include multi-task option too (train SPN on diff games in alternation)
