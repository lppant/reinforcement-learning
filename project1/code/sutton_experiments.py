from utils import *
from hyperparameters import *
from temporal_difference import calculate_td_value
import logging


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    experiment_figure3()
    experiment_figure4_and_figure5()
    return


def experiment_figure3():
    logging.debug('Experiments for Figure 3')

    # Different Lambdas
    lambda_set = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    avg_rmse = []
    training_sets = []
    initial_states_fig3 = get_initial_states()

    # Populate training sets
    for i in range(TRAINING_SETS):
        # Control error by giving maximum sequence size
        training_sets.append(get_sequences(initial_states_fig3, SEQUENCES, max_sequence_size=MAX_SEQUENCE_SIZE))

    # Calculate rmse
    for lambda_v in lambda_set:
        logging.debug('Executing TD(%s)', lambda_v)
        td_v_list = [calculate_td_value(sequences=training_set, ld_v=lambda_v, alpha=0.21, alpha_decay=0.9, threshold=1e-3) for training_set in training_sets]
        avg_rmse.append(calculate_avg_rmse(td_v_list, IDEAL_PREDICTIONS))

    logging.debug(avg_rmse)
    lambda_plot(lambda_set, avg_rmse, 'figure_3.png')


def experiment_figure4_and_figure5():
    logging.debug('Experiments for Figure 4')

    # Different Lambdas
    lambda_set = [0.0, 0.3, 0.8, 1.0]
    # Different Alphas
    alpha_set = np.arange(0, 0.6, 0.05)

    avg_rmse = []
    optimal_alphas = []
    training_sets = []
    initial_states_fig4 = get_initial_states()

    for i in range(TRAINING_SETS):
        # Control error by limiting sequence size
        training_sets.append(get_sequences(initial_states_fig4, SEQUENCES, max_sequence_size=MAX_SEQUENCE_SIZE))

    for lambda_v in lambda_set:
        logging.debug('Executing TD(%s)', lambda_v)
        lambda_errors = []
        optimal_alpha = 0

        # initialized to a very high value
        min_lambda_error = 100

        for a in alpha_set:
            td_v_list = [calculate_td_value(sequences=training_set, ld_v=lambda_v, alpha=a, max_iter=1) for training_set in training_sets]
            error = calculate_avg_rmse(td_v_list, IDEAL_PREDICTIONS)
            lambda_errors.append(error)
            if error < min_lambda_error:
                min_lambda_error = error
                optimal_alpha = a

        optimal_alphas.append(round(optimal_alpha, 2))
        avg_rmse.append(lambda_errors)

    if TEST_MODE:
        logging.debug(avg_rmse)
    alpha_plot(alpha_set, avg_rmse, 'figure_4.png', lambda_set)

    logging.debug('Experiments for Figure 5')

    assert (len(lambda_set) == len(optimal_alphas))

    avg_rmse = []

    for i in range(len(lambda_set)):
        logging.debug('Executing TD(%s)', lambda_set[i])
        td_v_list = [calculate_td_value(sequences=training_set, ld_v=lambda_set[i], alpha=optimal_alphas[i], max_iter=1) for training_set in training_sets]
        avg_rmse.append(calculate_avg_rmse(td_v_list, IDEAL_PREDICTIONS))

    if TEST_MODE:
        logging.debug(avg_rmse)

    lambda_plot(lambda_set, avg_rmse, 'figure_5.png')


if __name__ == '__main__':
    main()
