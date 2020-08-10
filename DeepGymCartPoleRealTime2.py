import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

# visualize data and weight/biases as well as accuracy over time
# Investigate ideal parameters and model structure


EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.2  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))  # returns x or y depending on condition

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, total_score):
        self.memory.append([state, action, reward, next_state, done, total_score])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train(self, batch_size):  # model fitting process
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, run_score in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    plt.show()  # To plot the scores
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("cartpole-ddqn.h5")
    done = False
    batch_size = 64
    run_score = 0
    start_of_append = 0
    score_min = 0  # Initializing, will increase this minimum score as model gets better
    deleted_elements = 0  # To keep track of numb of deleted elements

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time_lasted in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10  # less reward if we don't beat the round
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done, run_score)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, epsilon: {:.2}"
                      .format(e, EPISODES, time_lasted, agent.epsilon))
                # time_lasted is the number of states encountered (reward per state = 1)
                # Since the run is done, lets append the final scores to memory
                for item_count in range(start_of_append, start_of_append + time_lasted + 1 - deleted_elements):
                    agent.memory[item_count][5] = time_lasted  # Add score to complete run
                    print("Run numb: {}, done? {}, Score: {}".format(item_count, agent.memory[item_count][4],
                                                                     agent.memory[item_count][5]))
                start_of_append = start_of_append + time_lasted + 1 - deleted_elements
                break

            if len(agent.memory) > batch_size:
                agent.train(batch_size)

        print("LENGTH:", len(agent.memory))
        # plotting the score of the episode
        plt.scatter(e, time_lasted, color='red')
        plt.pause(0.1)
        #if e % 10 == 0:
        #    agent.save("cartpole-ddqn.h5")

plt.savefig('Cartpole_trial_original_bigbatch.png')
