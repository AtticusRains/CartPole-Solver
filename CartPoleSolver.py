import gym
import random
import numpy as np
import tflearn
from datetime import datetime
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from gym.wrappers.monitoring.video_recorder import VideoRecorder

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 1000
score_requirement = 50
initial_games = 10000
#capture = False

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range (goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            
            prev_observation = observation
            score += reward
            if done:
                break

        if score >+ score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Medain accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

initial_population()

def nural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name='input')
    
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = nural_network_model(input_size = len(x[0]))

    model.fit({'input':x}, {'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')
    return model

training_data = initial_population()
model = train_model(training_data)

#model.save(first.model)
#model.load()

def run_model(num_games, capture=False):

    scores = []
    choices = []

    for each_game in range(num_games):
        path = 'rec/' + str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        rec = VideoRecorder(env, base_path=path)
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            if(capture):
                rec.capture_frame()
            else:
                env.render()
            

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)
        rec.close()

    print('Average Score:', sum(scores)/len(scores))

run_model(10)