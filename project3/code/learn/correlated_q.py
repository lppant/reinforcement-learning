import numpy as np
import time

from cvxopt import matrix, solvers
from .soccer_game import SoccerGame
from .constants import *
from scipy.linalg import block_diag


def correlated_q_agent():
    print('Starting Correlated_Q')

    # discount rate
    gamma = 0.9

    epsilon_min = 0.001
    epsilon_decay = 10 ** (np.log10(epsilon_min) / STEPS_UPPER)


    # learning rate
    alpha, alpha_min = 1.0, 0.001
    alpha_decay = 10 ** (np.log10(alpha_min) / STEPS_UPPER)

    step_q_diffs = []

    np.random.seed(1234)

    # the state-action space depends on:
    # player A positions(8)
    # player B positions(8)
    # ball holder(2)
    # player A actions(5)
    # player B actions(5)
    # Q_tables of player A and player B
    Q_player_A = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q_player_B = np.ones((8, 8, 2, 5, 5)) * 1.0

    # state-value space depends:
    # player A positions(8)
    # player B positions(8)
    # ball holder(2)
    V_player_A = np.ones((8, 8, 2)) * 1.0
    V_player_B = np.ones((8, 8, 2)) * 1.0

    # starting joint policy for both players and split probability for each state by 25
    Pi = np.ones((8, 8, 2, 5, 5)) * 1 / 25

    # index for step
    counter = 0

    start_time = time.time()

    while counter < STEPS_UPPER:
        soccer = SoccerGame()
        # initial state
        s = [soccer.grid_position(0), soccer.grid_position(1), soccer.index_ball_holder]

        game_steps = 0
        done = 0

        while game_steps <= 100 and not done:
            if counter % 1000 == 0:
                print('\rStep {}, Alpha: {:.3f}, Time: {:.2f}'.format(counter, alpha, time.time() - start_time), end="")

            counter, game_steps = counter + 1, game_steps + 1

            # player A at state s moves south while player B sticks
            prev_Q = Q_player_A[2][1][1][2][4]

            actions = get_action(Pi, s, counter, epsilon_decay)

            sp, rewards, done = soccer.play_game(actions)

            alpha = alpha_decay ** counter

            # Compute Q-learning
            Q_player_A[s[0]][s[1]][s[2]][actions[0]][actions[1]] = (1 - alpha) * \
                                                                   Q_player_A[s[0]][s[1]][s[2]][actions[0]][
                                                                       actions[1]] + alpha * (rewards[0] + gamma *
                                                                                              V_player_A[sp[0]][sp[1]][
                                                                                                  sp[2]])

            Q_player_B[s[0]][s[1]][s[2]][actions[1]][actions[0]] = (1 - alpha) * \
                                                                   Q_player_B[s[0]][s[1]][s[2]][actions[1]][
                                                                       actions[0]] + alpha * (rewards[1] + gamma *
                                                                                              V_player_B[sp[0]][sp[1]][
                                                                                                  sp[2]].T)

            value_A, value_B, probability = lp_for_correlated_q(s, Q_player_A, Q_player_B)

            # update only if probability not null
            if probability is not None:
                V_player_A[s[0]][s[1]][s[2]] = value_A
                V_player_B[s[0]][s[1]][s[2]] = value_B
                Pi[s[0]][s[1]][s[2]] = probability

            s = sp

            # player A at state s moves south while player B sticks
            next_Q = Q_player_A[2][1][1][2][4]

            step_q_diffs.append(np.abs(next_Q - prev_Q))

    print('')
    return np.array(step_q_diffs)


def get_action(Pi, s, counter, epsilon_decay):
    # Take action with epsilon-greedy choice
    # decay epsilon
    epsilon = epsilon_decay ** counter

    if np.random.random() < epsilon:
        index = np.random.choice(np.arange(25), 1)
        # return the action as a 2-d index value
        return np.array([index // 5, index % 5]).reshape(2)

    else:
        # pi gives the probability of selecting an action
        index = np.random.choice(np.arange(25), 1, p=Pi[s[0]][s[1]][s[2]].reshape(25))
        # return the action as a 2-d index value
        return np.array([index // 5, index % 5]).reshape(2)


def block_diag_matrix(s, Q):
    # subset the selection condition for given Q
    Q_s = Q[s[0]][s[1]][s[2]]
    block_diag_matrix_val = block_diag(Q_s - Q_s[0, :], Q_s - Q_s[1, :], Q_s - Q_s[2, :],
                   Q_s - Q_s[3, :], Q_s - Q_s[4, :])
    return block_diag_matrix_val


def calc_probability(values):
    return np.abs(np.array(values).reshape((5, 5))) / sum(np.abs(values))


# Solving correlated-equilibrium with Linear Programming
def lp_for_correlated_q(s, Q_player_A, Q_player_B):

    # standard linear programming implementation used from cvxopt package

    # probability constraints
    A = matrix(np.ones((1, 25)))
    b = matrix(1.0)

    block_diag_mat_A = block_diag_matrix(s, Q_player_A)

    # row level indices in a list
    row_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
    paramA = block_diag_mat_A[row_indices, :]

    block_diag_mat_B = block_diag_matrix(s, Q_player_B)

    # column level indices in a list
    col_indices = [0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24]
    paramB = block_diag_mat_B[row_indices, :][:, col_indices]

    # rationality constraints
    G = matrix(np.append(np.append(paramA, paramB, axis=0), -np.eye(25), axis=0))
    h = matrix(np.zeros(65) * 0.0)

    c = matrix((Q_player_A[s[0]][s[1]][s[2]] + Q_player_B[s[0]][s[1]][s[2]].T).reshape(25))

    try:
        lp_solver = solvers.lp(c=c, G=G, h=h, A=A, b=b)
        if lp_solver['x'] is not None:
            probability = calc_probability(lp_solver['x'])
            value_player_A = np.sum(probability * Q_player_A[s[0]][s[1]][s[2]])
            value_player_B = np.sum(probability * Q_player_B[s[0]][s[1]][s[2]].T)
        else:
            value_player_A = None
            value_player_B = None
            probability = None
    except:
        value_player_A = None
        value_player_B = None
        probability = None

    return value_player_A, value_player_B, probability
