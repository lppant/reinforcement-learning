import string
import os
import numpy as np
import matplotlib.pyplot as plt
from hyperparameters import NON_TERMINAL_STATES, IDEAL_PREDICTIONS, OUTPUT_DIR
import logging


class RandomWalkState:
    def __init__(self, pos, name, is_terminal, reward=0.0, eligibility=0.0, value=0.0):
        self.pos = pos
        self.name = name
        self.is_terminal = is_terminal
        self.reward = reward
        self.eligibility = eligibility
        self.value = value


def get_initial_states():
    states = [RandomWalkState(0, 'A', is_terminal=True, reward=0.0, value=0.0)]
    for i in range(NON_TERMINAL_STATES):
        states.append(RandomWalkState(i + 1, string.ascii_uppercase[i + 1], is_terminal=False, reward=0.0, value=0.5))
    states.append(RandomWalkState(NON_TERMINAL_STATES + 1, 'G', is_terminal=True, reward=1.0, value=0.0))
    return states


def calculate_avg_rmse(estimated_value, actual_value):
    diff = np.subtract(estimated_value, actual_value)
    try:
        avg_rmse = np.mean(np.mean(pow(diff, 2), axis=1) ** 0.5)
    except IndexError:
        avg_rmse = np.mean(np.mean(pow(diff, 2)) ** 0.5)
    return avg_rmse


def get_sequences(states, sequence_count, max_sequence_size=8):
    sequences = []
    starting_pos = 3
    for i in range(sequence_count):
        pos = starting_pos
        sequence = [states[pos]]
        while True:
            if len(sequence) > max_sequence_size:
                pos = starting_pos
                sequence = [states[pos]]
            if sequence[len(sequence) - 1].is_terminal:
                break
            pos = get_neighbour_position(pos)
            sequence.append(states[pos])
        sequences.append(sequence)
    return sequences


def get_neighbour_position(pos):
    if np.random.random_integers(0, 1):
        return pos + 1
    else:
        return pos - 1


def lambda_plot(x, y, plot_name):
    fig = plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(u'$\\lambda$')
    plt.ylabel('ERROR')
    fig.savefig(os.path.join('..', 'out', OUTPUT_DIR, plot_name))
    logging.debug('plot created at ' + os.path.join('..', 'out', OUTPUT_DIR, plot_name))
    plt.close()
    # plt.show()


def alpha_plot(x, y, plot_name, ld_vals):
    fig = plt.figure()
    for i in range(len(y)):
        plt.plot(x, y[i], label='lambda=' + str(ld_vals[i]), marker='o')
    plt.xlim((-0.05, 0.65))
    plt.ylim((0.05, 0.75))
    plt.xlabel(u'$\\alpha$')
    plt.ylabel('ERROR')
    plt.legend(loc=2)
    fig.savefig(os.path.join('..', 'out', OUTPUT_DIR, plot_name))
    logging.debug('plot created at ' + os.path.join('..', 'out', OUTPUT_DIR, plot_name))
    plt.close()
    # plt.show()
