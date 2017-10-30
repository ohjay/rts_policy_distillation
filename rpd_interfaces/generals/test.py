#!/usr/bin/env python

import os
from random import randint, random
from rpd_interfaces.generals import generals

def run_random_agent():
    user_id = os.environ['GENERALS_UID']
    iface = generals.Generals(user_id)
    iface.join_game('1v1')

    for update in iface.get_updates():
        # Check if the game is over
        if update['result'] is not None:
            print(update['result'])
            break

        width, height = update['width'], update['height']
        size, terrain = update['size'], update['terrain']
        cities = update['cities']

        # Make a random move
        while True:
            # Pick a random tile
            start_index = randint(0, size - 1)

            # If we own the tile, make a random move starting from it
            if terrain[start_index] == iface.player_index:
                row = start_index // width
                col = start_index % width
                end_index = start_index

                rand = random()
                if rand < 0.25 and col > 0:
                    end_index -= 1  # left
                elif rand < 0.5 and col < width - 1:
                    end_index += 1  # right
                elif rand < 0.75 and row < height - 1:
                    end_index += width  # down
                elif row > 0:
                    end_index -= width
                else:
                    continue

                # Don't attack cities
                if end_index in cities:
                    continue

                iface.attack(start_index, end_index)
                break

if __name__ == '__main__':
    run_random_agent()
