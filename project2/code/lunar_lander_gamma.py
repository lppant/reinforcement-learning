import gym
import numpy as np
from collections import deque

from datetime import datetime
import os
import pickle
import logging

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
from agent.dqn import DQN


def training(gamma=0.99, soft_update_rate=1e-3, learning_rate=5e-3, epsilon_decay=0.995):
    logging.debug('Started training')

    # initialize the DQN env and agent
    env = gym.make("LunarLander-v2")
    learning_agent = DQN(env=env, gamma=gamma, learning_rate=learning_rate, soft_update_rate=soft_update_rate)

    # Max training episodes
    max_episodes = 1000

    # Max steps for each training episode
    max_steps = 1000

    # stores score for each episode
    scores = []

    # stores score for most recent 100 episodes
    relevant_scores = deque(maxlen=100)

    # epsilon and epsilon decay for epsilon-greedy action choice
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_list = []

    # for each episode
    for episode in range(max_episodes):
        epsilon_list.append(epsilon)
        curr_state = env.reset().reshape(1, 8)
        score = 0

        # each step in a training episode
        for step in range(max_steps):

            # get action from learning agent
            action = learning_agent.choose_action(curr_state, epsilon)
            # take action
            new_state, reward, done, info = env.step(action)
            score += reward
            new_state = new_state.reshape(1, 8)

            # save experience to replay memory
            learning_agent.update_replay_memory(curr_state, action, reward, new_state, done)
            # learn the model from current step
            learning_agent.learn()
            # update state
            curr_state = new_state

            if done:
                break

        scores.append(score)
        relevant_scores.append(score)
        avg_relevant_score = np.mean(relevant_scores)

        # apply epsilon decay
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)

        print('\rEpisode: {}, Average Score: {:.4f}, Epsilon: {:.4f}'.format(episode, avg_relevant_score, epsilon),
              end="")
        if episode % 50 == 0:
            print('\rEpisode: {}, Average Score: {:.4f}, Epsilon: {:.4f}'.format(episode, avg_relevant_score, epsilon))

        # if mean of last 100 scores above 200, then save model and stop training
        if avg_relevant_score >= 200.0:
            learning_agent.save("dqn_model.h5")
            logging.debug(
                'Required {:d} episodes to get average score of 200+ for 100 consecutive episodes. Average Score: {:.4f}'.format(
                    episode, avg_relevant_score))
            break

    env.close()
    return scores, epsilon_list


def plot_hyperparameters(scores, output_dir, graph_name, hp_set, hp_type, y_label="Scores"):
    fig = plt.figure()
    for i in range(len(scores)):
        plt.plot(np.arange(len(scores[i])), scores[i], label=hp_type + '=' + str(hp_set[i]))
    plt.xlabel('Episodes')
    plt.ylabel(y_label)
    plt.legend(loc=2)
    fig.savefig(os.path.join('..', 'out', output_dir, graph_name))
    logging.debug('plot created at ' + os.path.join('..', 'out', output_dir, graph_name))
    plt.close()


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # Create output directory
    output_root = '../out'
    output_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    os.mkdir(os.path.join(output_root, output_dir))
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Training for different gammas
    gamma_scores = []
    gamma_set = [0.99, 0.80, 0.60, 0.40]

    for gamma in gamma_set:
        print("Using gamma : {:.4f}\n".format(gamma), end="")
        gamma_score, _ = training(gamma=gamma)
        gamma_scores.append(gamma_score)
    plot_hyperparameters(gamma_scores, output_dir, "gamma_scores.png", gamma_set, "gamma")
    pickle.dump(gamma_scores, open("gamma_scores.pickle", "wb"))


if __name__ == '__main__':
    main()
