import numpy as np
from hyperparameters import *
from utils import get_initial_states
import logging


def calculate_td_value(sequences, ld_v, gamma=1.0, alpha=0.3, alpha_decay=0.9, max_iter=MAX_ITERATIONS,
                       threshold=0.001):
    states = get_initial_states()
    V = np.ndarray((0, NON_TERMINAL_STATES))

    has_converged = False
    itr = 0
    while not has_converged:
        value_delta = np.zeros(NON_TERMINAL_STATES)

        # Sequences depicts 1 training set
        for loc, sequence in enumerate(sequences):
            for state in states:
                state.eligibility = 0

            for t in range(1, len(sequence)):
                curr_state = states[sequence[t].pos]
                prev_state = states[sequence[t - 1].pos]

                prev_state.eligibility += 1
                state_delta = curr_state.reward + gamma * curr_state.value - prev_state.value

                for state in states:
                    update = alpha * state.eligibility * state_delta

                    if not state.is_terminal:
                        value_delta[state.pos - 1] += update
                    state.eligibility *= ld_v * gamma

            # Update alpha (learning rate)
            alpha = alpha * alpha_decay

        itr += 1

        # Apply weight update after each presentation of TRAINING SET
        for i in range(len(value_delta)):
            states[i + 1].value += value_delta[i]

        V = np.vstack([V, [states[i].value for i in range(1, len(states) - 1)]])

        if itr >= max_iter:
            if TEST_MODE:
                logging.debug('Max iterations limit reached')
            break

        dist = np.sqrt(np.sum([pow(dv, 2) for dv in value_delta]))
        if dist < threshold:
            has_converged = True

    return V[len(V) - 1]
