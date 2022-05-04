from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np
import pickle
import copy
import argparse
import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# General Parameters
MAX_STEPS = 75

WAREHOUSE_ART = \
    ['    ', 
     'GP  ',
     '    ',
     '    ']

BACKGROUND_ART = \
    ['    ',
     '    ',
     '    ',
     '    ']


WAREHOUSE_FG_COLOURS = {' ': (870, 838, 678),  # Floor.
                        'P': (388, 400, 999),  # The player.
                        'G': (900, 300, 900),      # Goal.
                        'x': (850, 603, 270),  # Unused.
                        }


def make_game(seed=None, demo=False):
    warehouse_art = WAREHOUSE_ART
    what_lies_beneath = BACKGROUND_ART
    sprites = {'P': PlayerSprite}

    if demo:
        raise NotImplementedError
    else:
        drapes = {'X': JudgeDrape}

    drapes['G'] = GoalDrape

    update_schedule = [['G'],
                       ['P'],
                       ['X']]


    return ascii_art.ascii_art_to_game(
        warehouse_art, what_lies_beneath, sprites, drapes,
        update_schedule=update_schedule)


def scalar_to_idx(x):
    print(x)
    row = x%4
    col = int(np.floor(x/4))
    return (row, col)


class GoalDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop  # Unused.

        player_pattern_position = things['P'].position
        player_row = player_pattern_position.row
        player_col = player_pattern_position.col

        if self.curtain[(player_row, player_col)]:
            the_plot.add_reward(0.1)

class PlayerSprite(prefab_sprites.MazeWalker):

    def __init__(self, corner, position, character):
        """Constructor: simply supplies characters that players can't traverse."""
        super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#H.')

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things, layers  # Unused.

        if the_plot.frame == 0:
            print(np.random.randint(4))
            self._teleport(scalar_to_idx(np.random.randint(4)))

        if actions == 0:    # go upward?
            self._north(board, the_plot)
        elif actions == 1:  # go downward?
            self._south(board, the_plot)
        elif actions == 2:  # go leftward?
            self._west(board, the_plot)
        elif actions == 3:  # go rightward?
            self._east(board, the_plot)


class JudgeDrape(plab_things.Drape):
    def __init__(self, curtain, character):
        super(JudgeDrape, self).__init__(curtain, character)
        self._step_counter = 0
        self._max_steps = MAX_STEPS

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Clear our curtain and mark the locations of all the boxes True.
        self.curtain.fill(False)
        the_plot.add_reward(0)
        #the_plot.add_reward(-0.1)
        self._step_counter += 1

        # See if we should quit: it happens if the user solves the puzzle or if
        # they give up and execute the 'quit' action.
        if (actions == 4) or (self._step_counter == self._max_steps):
            the_plot.terminate_episode()


def main(demo):
    # Builds an interactive game session.
    game = make_game(demo=demo)


    ui = human_ui.CursesUi(
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         'w': 5,
                         's': 6,
                         'a': 7,
                         'd': 8,
                         -1: 4,
                         'q': 9, 'Q': 9},
        delay=1000,
        colour_fg=WAREHOUSE_FG_COLOURS)

    # Let the game begin!
    ui.play(game)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo', help='Record demonstrations',
                        action='store_true')
    args = parser.parse_args()
    if args.demo:
        main(demo=True)
    else:
        main(demo=False)
