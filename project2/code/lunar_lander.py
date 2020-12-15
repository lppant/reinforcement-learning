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
    max_episodes = 2000

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
            learning_agent.save("dqn_model_train.h5")
            logging.debug(
                'Required {:d} episodes to get average score of 200+ for 100 consecutive episodes. Average Score: {:.4f}'.format(
                    episode, avg_relevant_score))
            break

    env.close()
    return scores, epsilon_list


def test(model_file):
    logging.debug('Started testing')
    # Load trained agent
    model = load_model(model_file)
    model.summary()

    # plot model structure
    plot_model(model, show_shapes=True, to_file='model.png')
    logging.debug("model weights:", model.get_weights())

    # test the trained agent
    env = gym.make("LunarLander-v2")
    scores = []

    # run the agent for 100 episodes
    for episode in range(100):
        cur_state = env.reset().reshape(1, 8)
        score = 0
        for step in range(1000):
            action = np.argmax(model.predict(cur_state)[0])
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 8)
            score += reward
            cur_state = new_state
            if done:
                break

        scores.append(score)
        print('\rEpisode: {}, Score: {:.4f}'.format(episode, score), end="")

    env.close()
    return scores


def plot_graph(scores, output_dir, graph_name, y_label="Scores"):
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Episodes')
    plt.ylabel(y_label)
    fig.savefig(os.path.join('..', 'out', output_dir, graph_name))
    logging.debug('Graph created at ' + os.path.join('..', 'out', output_dir, graph_name))
    # plt.show()


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # Create output directory
    output_root = '../out'
    output_dir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(output_root):
        os.mkdir(output_root)
    os.mkdir(os.path.join(output_root, output_dir))
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # train and plot using best parameters set as default
    training_scores, epsilons = training()
    plot_graph(training_scores, output_dir, "train_scores.png")
    plot_graph(epsilons, output_dir, "epsilons.png", "Epsilons")
    pickle.dump(training_scores, open("train_scores.pickle", "wb"))
    pickle.dump(epsilons, open("epsilons.pickle", "wb"))

    # test using trained model
    test_scores = test("dqn_model_train.h5")
    plot_graph(test_scores, output_dir, "test_scores.png")
    pickle.dump(test_scores, open("test_scores.pickle", "wb"))


if __name__ == '__main__':
    main()
