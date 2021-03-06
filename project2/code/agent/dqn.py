from collections import deque
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQN:
    def __init__(self, env, gamma=0.99, soft_update_rate=1e-3, learning_rate=5e-4):
        self.env = env
        self.gamma = gamma
        self.soft_update_rate = soft_update_rate
        self.learning_rate = learning_rate

        # Use deque for replay memory as it looks back only upto `maxlen` elements.
        self.replay_memory = deque(maxlen=int(1e5))

        # model for action-value function
        self.model = self.initialize()

        # model for target action-value function
        self.target_model = self.initialize()

        # batch size
        self.batch_size = 64

        # step counter
        self.step_counter = 0

        # Soft Update target model Q_hat to model Q after every few steps
        self.update_freq = 4

    # create keras based neural network
    def initialize(self):
        # input shape is state space (8,1)
        input_shape = self.env.observation_space.shape[0]

        # output shape is actions space (4,1)
        output_shape = self.env.action_space.n

        model = Sequential()

        # Add 2 hidden layers of size 64
        model.add(Dense(64, input_dim=input_shape, activation="relu"))
        model.add(Dense(64, activation="relu"))
        # Add output layer
        model.add(Dense(output_shape))

        # Use MSE as loss function
        # Use Adam optimizer
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def update_replay_memory(self, curr_state, action, reward, new_state, done):
        self.replay_memory.append([curr_state, action, reward, new_state, done])

    def choose_action(self, state, epsilon):
        # Take action with epsilon-greedy choice
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        predicted_value = self.model.predict(state)
        return np.argmax(predicted_value[0])

    # gradient descent implementation for the model
    def learn(self):

        # update counter
        self.step_counter += 1
        self.step_counter %= self.update_freq

        # make sure that replay memory is filled at least up to batch size
        if self.step_counter == 0:
            if len(self.replay_memory) < self.batch_size:
                return

            # init list states to store states
            states = []
            # init list of targets values prediction generated by model Q associated with each state-action
            targets_prediction = []

            # pick random samples from replay memory
            samples = random.sample(self.replay_memory, self.batch_size)

            for state, action, reward, new_state, done in samples:
                if done:
                    target = reward
                else:
                    q_new_state = np.amax(self.target_model.predict(new_state)[0])
                    target = reward + self.gamma * q_new_state

                target_prediction = self.model.predict(state)
                target_prediction[0][action] = target

                states.append(state[0])
                targets_prediction.append(target_prediction[0])

            # fit the model
            self.model.fit(np.array(states), np.array(targets_prediction), epochs=1, verbose=0)

            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            # soft update to target model Q_hat from model Q
            for i in range(len(target_weights)):
                target_weights[i] = self.soft_update_rate * weights[i] + (1 - self.soft_update_rate) * target_weights[i]
            self.target_model.set_weights(target_weights)

    # save model
    def save(self, filepath):
        self.model.save(filepath)
