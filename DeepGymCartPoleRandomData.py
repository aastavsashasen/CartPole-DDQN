import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v1')  # create the environment
env.reset()
goal_steps = 500
score_requirement = 90
initial_games = 10000

# we now want to populate with data
# to get our training data we take random steps and record game results in list
def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(initial_games):  # we play 10000 times to collect data
        score = 0
        game_memory = []  # previous observation + action
        previous_observation = []  # previous cart and pole observation
        for step_index in range(goal_steps):  # play for 500 steps, which means completing the game
            # env.render()  # if you want to see the games being played
            action = random.randrange(0, 2)  # random action for each step (0 or 1)
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:  # if its not the first step, save it
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward  # in this game, surviving gives you a reward
            if done:  # if its over before 500 steps, end loop
                break

        if score >= score_requirement:  # if we played more than 60 steps, filters out garbage runs
            accepted_scores.append(score)
            for data in game_memory:  # contains observation, action
                if data[1] == 1:  # takes action
                    output = [0, 1]  # one hot for right
                elif data[1] == 0:
                    output = [1, 0]  # one hot for left
                training_data.append([data[0], output])  # observation (float), left/right (one hot)

        env.reset()

    print(accepted_scores)
    return training_data

# sample of OpenAI implementation:
# this is how we play the game for 1000 steps
#for step_index in range(1000):  # playing in steps
#    env.render()  # render the game so it runs in the loop
#    action = env.action_space.sample()  # gets a random action
#    observation, reward, done, info = env.step(action)  # doing that random action
#    print("Step {}:".format(step_index))  # function in which we can do an action
#    print("action: {}".format(action))  # 0 means left, 1 means right
#    print("observation: {}".format(observation))  # what the game returns, what is seen
#    print("reward: {}".format(reward))  # reward from previous action
#    print("done: {}".format(done))  # if its time to reset env
#    print("info: {}".format(info))
#    if done:
#        break

# now to build our neural network
# consists of dense layers, essentially moves in direction of pole falling
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    # relu for neurons, softmax for classification (%prob), linear for regression (value predict)
    model.compile(loss='mse', optimizer=Adam())

    return model

def train_model(training_data):
    # reshape (-1) means flatten out in single row or single col (-1,1)
    # essentially means make compatible with previous shape
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    # take all the observations and put it in groups of 4 in each row (len(training_data[0][0]))
    # X holds observations, Y holds actions
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    # take all the actions and put it in groups of 2 in each row (len(training_data[0][1]))
    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    # input observed, output action
    # then finally to train...
    model.fit(X, y, epochs=20)
    # we relate each observation to the action
    return model

# finally train our model on the collected random training data
trained_model = train_model(training_data=model_data_preparation())

# now lets play a game with this bot
scores = []  # stores what scores we got
choices = []  # and what choices we made
for each_game in range(100):  # play 100 games
    score = 0
    prev_obs = []  # previous observation
    for step_index in range(goal_steps):  # take only 500 steps per game
        env.render()
        if len(prev_obs) == 0:  # first step take random
            action = random.randrange(0, 2)
        else:  # here we use our model to take a step (max prob)
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score += reward
        if done:
            break
    env.reset()
    scores.append(score)

print(scores)
print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))

env.close()
