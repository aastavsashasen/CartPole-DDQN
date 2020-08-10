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
# train only on good data (higher score data) as time progresses

EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.90    # discount rate for future rewards
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
        # model.add(Dense(24, activation='relu'))
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
    batch_size = 32
    run_score = 0
    start_of_append = 0
    score_list = []
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
                if not (time_lasted in score_list):  # So we do not repeat scores in score list
                    score_list.append(time_lasted)

                for index, val in enumerate(agent.memory):
                    if val[5] == 0:
                        agent.memory[index][5] = time_lasted  # appending final score to run states/frames

                print("Memory length:", len(agent.memory))
                agent_scores = [row[5] for row in agent.memory]
                print("Agent scores:", agent_scores)
                print("Minimum in agent scores list = ", min(agent_scores))
                print("Score list:", score_list)

                if (len(agent.memory) > 1000) and (min(agent_scores) < 200):
                    min_score_value = min(score_list)
                    print("lowest score to remove = ", min_score_value)
                    for index2, val2 in enumerate(agent.memory):
                        if val2[5] == min_score_value or val2[5] == 0:  # Check all places of min_score_value or 0
                            for i in range(-1, min_score_value):  # delete all min_score_value(s) by iterating delete for the # of times they occur in the place they start
                                del agent.memory[index2]  # Remember that deleting changes all indices, so we can stay on the same index and keep deleting as the further indices move down
                    # Ensure no min_score_value(s) left, if so delete individually
                    agent_scores = [row[5] for row in agent.memory]  # recalculate
                    while min(agent_scores) == min_score_value:  # delete leftover scores due to concatenating error
                        for index3, val3 in enumerate(agent.memory):
                            if val3[5] == min_score_value:
                                del agent.memory[index3]
                        agent_scores = [row[5] for row in agent.memory]  # recalculate
                    score_list.remove(min_score_value)  # also remove from score list
                break

            if len(agent.memory) > batch_size:
                agent.train(batch_size)
            # ===============================================================================
            # still problem with not all scores being deleted:
            # delete individually after the loop... ?
            # values missed sometimes upon repeated scores being placed next to each other by deletion of score in btw
            # ================================================================================

        print("Final memory LENGTH:", len(agent.memory))
        print("Reward table = ", [row[2] for row in agent.memory])
        # plotting the score of the episode
        plt.scatter(e, time_lasted, color='red')
        plt.pause(0.1)
        #if e % 10 == 0:
        #    agent.save("cartpole-ddqn.h5")


plt.savefig('Cartpole_trial.png')
