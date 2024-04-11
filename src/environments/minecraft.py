# minecraft environment
from src.environments.SlipperyGrid import SlipperyGrid
import numpy as np
import random

# create a SlipperyGrid object
minecraft = SlipperyGrid(shape=[10, 10], initial_state=[5, 2], slip_probability=0.05)


def reset(self):
    # define the labellings
    ## labels stochasticity
    rand_gen = random.random()
    if rand_gen < 0.05:
        i = 1
    elif 0.05 <= rand_gen < 0.1:
        i = -1
    else:
        i = 0

    # labels layout 1
    labels = np.empty([minecraft.shape[0], minecraft.shape[1]], dtype=object)
    labels[0:10, 0:10] = 'safe'
    labels[0:3, 5] = 'basalt_lava'
    labels[2, 8:10] = 'basalt_lava'
    labels[0][0] = labels[4 + i][5] = 'creeper'
    labels[2][2 + i] = labels[7][3 + i] = labels[5 + i][7] = 'wood'
    labels[0][3 + i] = labels[4 + i][0] = labels[9][4 + i] = 'iron'
    labels[6 + i][1] = labels[6][5 + i] = labels[4 + i][9] = 'work_bench'
    labels[2 + i][4] = labels[9][0] = labels[7][7+i] = 'smith_table'
    labels[0][7] = 'diamond'

    # ## labels layout 2
    # labels = np.empty([minecraft.shape[0], minecraft.shape[1]], dtype=object)
    # labels[0:10, 0:10] = 'safe'
    # labels[0:3, 3] = 'basalt_lava'
    # labels[6:, 3] = 'basalt_lava'
    # labels[0][0] = labels[4 + i][5] = 'iron'
    # labels[2][2 + i] = labels[7][3 + i] = labels[5 + i][7] = 'wood'
    # labels[0][3 + i] = labels[4 + i][0] = labels[9][4 + i] = 'creeper'
    # labels[6 + i][1] = labels[6][5 + i] = labels[4 + i][9] = 'work_bench'
    # labels[2 + i][4] = labels[9][0] = labels[7][7 + i] = 'smith_table'
    # labels[0][7] = 'diamond'

    # labels layout 3
    # labels = np.empty([minecraft.shape[0], minecraft.shape[1]], dtype=object)
    # labels[0:10, 0:10] = 'safe'
    # labels[0:4, 2] = 'basalt_lava'
    # labels[0:6, 6] = 'basalt_lava'
    # labels[0][0] = labels[4 + i][5] = 'creeper'
    # labels[2][2 + i] = labels[7][3 + i] = labels[5 + i][7] = 'wood'
    # labels[0][3 + i] = labels[4 + i][0] = labels[9][4 + i] = 'iron'
    # labels[6 + i][1] = labels[6][5 + i] = labels[4 + i][9] = 'work_bench'
    # labels[2 + i][4] = labels[9][0] = labels[7][7 + i] = 'smith_table'
    # labels[0][7] = 'diamond'

    # override the labels
    self.labels = labels
    self.current_state = self.initial_state.copy()


SlipperyGrid.reset = reset.__get__(minecraft, SlipperyGrid)
