import numpy as np
import time

from cvxopt import matrix, solvers
from .soccer_game import SoccerGame
from .constants import *


def foe_q_agent():
    print('Starting Foe_Q')

    def compute_q_pi_v(s, sp, Q_player_A, Q_player_B, Pi_player_A, Pi_player_B, V_player_A, V_player_B):
        Q_player_A[s[0]][s[1]][s[2]][actions[1]][actions[0]] = (1 - alpha) * \
                                                               Q_player_A[s[0]][s[1]][s[2]][actions[1]][
                                                                   actions[0]] + alpha * (rewards[0] + gamma *
                                                                                          V_player_A[sp[0]][
                                                                                              sp[1]][
                                                                                              sp[2]])

        value_A, prob_A = minimax(s, Q_player_A)
        V_player_A[s[0]][s[1]][s[2]] = value_A
        Pi_player_A[s[0]][s[1]][s[2]] = prob_A

        Q_player_B[s[0]][s[1]][s[2]][actions[0]][actions[1]] = (1 - alpha) * \
                                                               Q_player_B[s[0]][s[1]][s[2]][actions[0]][
                                                                   actions[1]] + alpha * (rewards[1] + gamma *
                                                                                          V_player_B[sp[0]][
                                                                                              sp[1]][
                                                                                              sp[2]])

        value_B, prob_B = minimax(s, Q_player_B)
        V_player_B[s[0]][s[1]][s[2]] = value_B
        Pi_player_B[s[0]][s[1]][s[2]] = prob_B

        return Q_player_A[2][1][1][4][2]

    # discount factor
    gamma = 0.9

    # learning rate
    alpha, alpha_min = 1.0, 0.001
    alpha_decay = 10 ** (np.log10(alpha_min) / STEPS_UPPER)

    # epsilon
    epsilon_min = 0.001
    epsilon_decay = 10 ** (np.log10(epsilon_min) / STEPS_UPPER)

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

    # the state-value space depends:
    # player A positions(8)
    # player B positions(8)
    # ball holder(2)
    V_player_A = np.ones((8, 8, 2)) * 1.0
    V_player_B = np.ones((8, 8, 2)) * 1.0

    # starting policy for player A and B, with equal probability for each action
    Pi_player_A = np.ones((8, 8, 2, 5)) * 1 / 5
    Pi_player_B = np.ones((8, 8, 2, 5)) * 1 / 5

    # index for step
    counter = 0

    start_time = time.time()

    while counter < STEPS_UPPER:
        soccer = SoccerGame()
        # initial state
        s = [soccer.grid_position(0), soccer.grid_position(1), soccer.index_ball_holder]

        done = 0
        while not done:
            if counter % 1000 == 0:
                print('\rStep {}, Alpha: {:.3f}, Time: {:.2f}'.format(counter, alpha, time.time() - start_time), end="")
            counter += 1

            # player A at state s moves south while player B sticks
            prev_Q = Q_player_A[2][1][1][4][2]

            actions = [get_action(s, counter, epsilon_decay, Pi_player_A),
                       get_action(s, counter, epsilon_decay, Pi_player_B)]
            sp, rewards, done = soccer.play_game(actions)

            # Compute Q-learning
            next_Q = compute_q_pi_v(s, sp, Q_player_A, Q_player_B, Pi_player_A, Pi_player_B, V_player_A, V_player_B)

            s = sp
            step_q_diffs.append(np.abs(next_Q - prev_Q))

            # decay learning rate
            alpha = alpha_decay ** counter

    print('')
    return np.array(step_q_diffs)


def calc_probability(values):
    return np.abs(values).reshape((5,)) / sum(np.abs(values))


def get_action(s, counter, epsilon_decay, pi):
    # Take action with epsilon-greedy choice
    # decay epsilon
    epsilon = epsilon_decay ** counter
    if np.random.random() < epsilon:
        return np.random.randint(0, 5)
    else:
        # pi gives the probability of selecting an action
        return np.random.choice([0, 1, 2, 3, 4], 1, p=pi[s[0]][s[1]][s[2]])[0]


# minimax is solved using LP
def minimax(s, Q):
    c = matrix([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    A = matrix([[0.0], [1.0], [1.0], [1.0], [1.0], [1.0]])
    b = matrix(1.0)
    G = matrix(np.append(
        np.append(np.ones((5, 1)), -Q[s[0]][s[1]][s[2]], axis=1),
        np.append(np.zeros((5, 1)), -np.eye(5), axis=1), axis=0))
    h = matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    lp_solver = solvers.lp(c=c, G=G, h=h, A=A, b=b)
    return np.array(lp_solver['x'][0]), calc_probability(lp_solver['x'][1:])
