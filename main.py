#!/usr/bin/env python

import os
import yaml
import argparse

SUPPORTED_ENVS = {'generals',}
SUPPORTED_OPS = {'train',}

def train(iface):
    pass

def test(iface):
    for update in iface.get_updates():
        observation = iface.extract_observation(update)

def run(config):
    """Run RPD code according to the passed-in config file."""
    assert config['env'] in SUPPORTED_ENVS, 'not a supported environment'
    assert config['operation'] in SUPPORTED_OPS, 'not a supported op'

    iface = None
    env_info = config.get(config['env'], None)
    if config['env'] == 'generals':
        from rpd_interfaces.generals import generals
        user_id = env_info['user_id']
        iface = generals.Generals(user_id)
        iface.join_game(env_info['mode'])

    eval(config['operation'])(iface)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, help='config path')
    args = parser.parse_args()

    assert os.path.isfile(args.config)
    config = yaml.load(open(args.config, 'r'))
    run(config)
