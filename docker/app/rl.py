from gcp_api_wrapper import shutdown, download_blob 
from google.cloud import storage 
import os 
from time import sleep 
import gym
import random
import pickle
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

EPISODES = 50000

EMBEDDING_DIM = 10 

data_path = '/dat/data.pkl'

if not os.path.isfile(data_path):
    download_blob('memory.pkl-backup', data_path)
    pass

class DQNAgent:
    def __init__(self, action_size, load_model=False, load_data=False):
        self.render = False
        self.load_model = load_model
        self.load_data = load_data 
        # environment settings
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 2000
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("/dat/breakout_dqn.h5")

        if self.load_data: 
            with open(data_path, 'rb') as f:
                self.memory = pickle.load(f) 

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(EMBEDDING_DIM)) 
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(Dense(self.action_size)) 
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, name):
        self.model.save_weights(name)

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def simulate(self, max_steps=None, viz=None, epsilon=0.): 
        '''
        args: 
        - `viz`: gameplay visualization settings 
          - If `None`, no gameplay will be visualized. 
          - If `jupyter`, images will be rendered for a Jupyter Server. 
        - `epsilon`: probability of random behavior 
        '''
        # initialise 
        self.epsilon = epsilon   
        env = gym.make('BreakoutDeterministic-v4') 
        observe = env.reset() 
        # agent as a memory of 4 steps 
        state = pre_processing(observe) 
        history = np.stack((state, state, state, state), axis=2) 
        history = np.reshape([history], (1, 84, 84, 4)) 
        if viz == 'jupyter': 
            plt.figure(figsize=(9,9))
            img = plt.imshow(env.render(mode='rgb_array')) # only call this once 
        wait_steps = random.randint(1, self.no_op_steps) # start game idle 
        step = 0 
        score = 0. 
        done = False 
        while not done: 
            if viz == 'jupyter': 
                img.set_data(env.render(mode='rgb_array')) # just update the data
                display.display(plt.gcf())
                display.clear_output(wait=True)
            # get action 
            if step < wait_steps:
                # early action is idle 
                action = 1
            else:
                action = self.get_action(history) 
            if action == 0:
                action = 1
            elif action == 1:
                action = 2
            else:
                action = 3
            # increment 
            observe, reward, done, info = env.step(action)
            step += 1 
            # rewards are clipped 
            reward = np.clip(reward, -1., 1.) 
            score += reward 
            # update agent memory  
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            history = np.append(next_state, history[:, :, :, :3], axis=3) 
            # is done? 
            if max_steps is not None:
                if step > max_steps:
                    done = True
        return score 

# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # login to storage 
    storage_client = storage.Client.from_service_account_json('/app/service-account.json') 
    bucket = storage_client.bucket(os.environ['BUCKET_NAME']) 
    agent = DQNAgent(action_size=3, load_data=True)
    for e in range(EPISODES):
        agent.train_replay() 
        if e % 10 == 0 and e > 0: 
            print(str(e+1)+'/'+str(EPISODES)+' ('+str(100.*(e+1)/EPISODES)+'%)') 
        if e % 1000 == 0 and e > 0:
            agent.model.save_weights("/dat/breakout_dqn.h5")
            # upload to GCP storage
            blob = bucket.blob('rl-full.h5')
            blob.upload_from_filename('/dat/breakout_dqn.h5')
            ## reporting 
            print('Simulation score: '+str(agent.simulate())) 
    # work complete
    while True:
        shutdown()
        sleep(100)
