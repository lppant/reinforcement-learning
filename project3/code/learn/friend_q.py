import numpy as np
import time

from .soccer_game import SoccerGame
from .constants import *


def friend_q_agent():
    print('Starting Friend_Q')

    def compute_q(s, sp, done, Q_player_A, Q_player_B):
        if done:
            Q_player_A[s[0]][s[1]][s[2]][actions[1]][actions[0]] = \
                Q_player_A[s[0]][s[1]][s[2]][actions[1]][actions[0]] + alpha * (
                        rewards[0] - Q_player_A[s[0]][s[1]][s[2]][actions[1]][actions[0]])

            Q_player_B[s[0]][s[1]][s[2]][actions[0]][actions[1]] = \
                Q_player_B[s[0]][s[1]][s[2]][actions[0]][actions[1]] + alpha * (
                        rewards[1] - Q_player_B[s[0]][s[1]][s[2]][actions[0]][actions[1]])
            # player A at state s moves south while player B sticks
            return Q_player_A[2][1][1][4][2]

        else:
            Q_player_A[s[0]][s[1]][s[2]][actions[1]][actions[0]] = \
                Q_player_A[s[0]][s[1]][s[2]][actions[1]][actions[0]] + alpha * (
                        rewards[0] + gamma * np.max(Q_player_A[sp[0]][sp[1]][sp[2]]) -
                        Q_player_A[s[0]][s[1]][s[2]][actions[1]][actions[0]])

            Q_player_B[s[0]][s[1]][s[2]][actions[0]][actions[1]] = \
                Q_player_B[s[0]][s[1]][s[2]][actions[0]][actions[1]] + alpha * (
                        rewards[1] + gamma * np.max(Q_player_B[sp[0]][sp[1]][sp[2]]) -
                        Q_player_B[s[0]][s[1]][s[2]][actions[0]][actions[1]])
            # player A at state s moves south while player B sticks
            return Q_player_A[2][1][1][4][2]

    # discount factor
    gamma = 0.9

    # Define the epsilon and its decay for epsilon-greedy action selection
    epsilon, epsilon_decay, epsilon_min = 0.5, 0.99995, 0.001

    # learning rate
    alpha, alpha_decay, alpha_min = 0.45, 0.99995, 0.001

    step_q_diffs = []

    np.random.seed(1)

    # the state-action space depends on:
    # player A positions(8)
    # player B positions(8)
    # ball holder(2)
    # player B actions(5)
    # player A actions(5)
    # Q_tables of player A and player B
    Q_player_A = np.zeros((8, 8, 2, 5, 5))
    Q_player_B = np.zeros((8, 8, 2, 5, 5))

    # index for step
    counter = 0

    start_time = time.time()

    while counter < STEPS_UPPER:
        soccer = SoccerGame()
        # initial state
        s = [soccer.grid_position(0), soccer.grid_position(1), soccer.index_ball_holder]

        while True:

            if counter % 1000 == 0:
                print('\rStep {}, Alpha: {:.3f}, Epsilon: {:.3f}, Time: {:.2f}'.format(counter, alpha, epsilon, time.time() - start_time), end="")

            # player A at state s moves south while player B sticks
            prev_Q = Q_player_A[2][1][1][4][2]

            actions = [get_action(Q_player_A, s, epsilon), get_action(Q_player_B, s, epsilon)]
            sp, rewards, done = soccer.play_game(actions)

            counter += 1

            # Compute Q-learning
            next_Q = compute_q(s, sp, done, Q_player_A, Q_player_B)

            step_q_diffs.append(abs(next_Q - prev_Q))

            if done:
                break
            else:
                s = sp

        alpha = alpha * alpha_decay
        if alpha < alpha_min:
            alpha = alpha_min
        epsilon = epsilon * epsilon_decay
        if epsilon < epsilon_min:
            epsilon = epsilon_min

    print('')
    return np.array(step_q_diffs)


def get_action(Q, s, epsilon):
    # Take action with epsilon-greedy choice
    if np.random.random() < epsilon:
        return np.random.randint(0, 5)
    else:
        arg_max = np.where(Q[s[0]][s[1]][s[2]] == np.max(Q[s[0]][s[1]][s[2]]))
        return arg_max[1][np.random.choice(range(len(arg_max[0])), 1)[0]]
