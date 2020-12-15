import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import os
import logging

from cvxopt import solvers
from learn import q_learning_agent, friend_q_agent, foe_q_agent, correlated_q_agent


def plot_graph(errors, output_dir, title):
    fig = plt.figure()
    plt.plot(errors, linestyle='-', linewidth=0.6)
    plt.title(title)
    plt.ylim(0, 0.5)
    # plt.xlim(0, 10**6)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value Difference')
    plt.ticklabel_format(style='sci', axis='x',
                         scilimits=(0, 0), useMathText=True)
    fig.savefig(os.path.join('..', 'out', output_dir, title + '.png'))
    logging.info('Plot created at ' + os.path.join('..', 'out', output_dir, title + '.png'))
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    solvers.options['show_progress'] = False
    plt.rcParams['agg.path.chunksize'] = 10000

    # Create output directory
    output_root = '../out'
    output_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    os.mkdir(os.path.join(output_root, output_dir))

    q_learning_q_diffs = q_learning_agent()
    plot_graph(q_learning_q_diffs, output_dir, 'Q-learner')

    friend_q_diffs = friend_q_agent()
    plot_graph(friend_q_diffs, output_dir, 'Friend-Q')

    foe_q_diffs = foe_q_agent()
    plot_graph(foe_q_diffs, output_dir, 'Foe-Q')

    correlated_q_diffs = correlated_q_agent()
    plot_graph(correlated_q_diffs, output_dir, 'Correlated-Q')


if __name__ == '__main__':
    main()
