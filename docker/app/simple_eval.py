
# Before running more-advanced experiments, I want to test for feasibility. 
# In an advanced evaluation, sampling should reflect the automaton's 
# experience, with the GAN being trained on only the observed real data. 
# In this simple evaluation, the GAN will not be trained--I need to 
# justify the engineering work first. The GAN is trained on a single, 
# large dataset. The evaluation will work as follows.
# 1. Sample data, pooling simulants with real. 
# 2. Fit the q-net 
# 3. Generate the metric: average score 

## libraries 
from gcp_api_wrapper import download_blob, upload_blob, shutdown
from cvae import CVAE
from rl import DQNAgent 
from transfer_sampler import inverse_transfer_sample
from keras.layers.advanced_activations import ReLU
from keras.layers import Dense 
from keras.models import Sequential, Model 
from keras.optimizers import RMSprop
from keras import backend as K 
import os 
import gym
import pickle
import random 
import numpy as np 
from time import time 

## constants 
CVAE_DATA_PATH='/dat/cvae-data.pkl'
CVAE_DATA_BLOB_NAME='cvae-data.pkl'
CVAE_MODEL_PATH='/dat/cvae-model.h5'
CVAE_MODEL_BLOB_NAME='cvae-model.h5'
FULL_RL_MODEL_PATH='/dat/breakout_dqn.h5'  
FULL_RL_MODEL_BLOB_NAME='rl-full.h5'

# ensure files are downloaded 
if not os.path.isfile(CVAE_DATA_PATH): 
    download_blob(CVAE_DATA_BLOB_NAME, CVAE_DATA_PATH) 
if not os.path.isfile(CVAE_MODEL_PATH): 
    download_blob(CVAE_MODEL_BLOB_NAME, CVAE_MODEL_PATH) 
if not os.path.isfile(FULL_RL_MODEL_PATH):
    download_blob(FULL_RL_MODEL_BLOB_NAME, FULL_RL_MODEL_PATH) 

# load files 
with open(CVAE_DATA_PATH, 'rb') as f: 
    CVAE_DATA = pickle.load(f)
CVAE_MODEL = CVAE(data_dim=512*2, label_dim=9, model_path=CVAE_MODEL_PATH)
FULL_RL_MODEL = DQNAgent(action_size=3, load_model=True) 

def simple_sample(n_real, n_fake): 
    '''
    Generates a mixed dataset of simulated and real embedded samples. 
    Samples are "embedded" because we've used transfer learning. 
    Sampling is "simple" because the GAN is not fit with each simple. 
    '''
    ## sample real data 
    real_data = [] 
    if n_real > 0:
        real_data = __sample_real_data(n_real) 
    ## sample fake data 
    fake_data = [] 
    if n_fake > 0: 
        fake_data = __sample_fake_data(n_fake) 
    ## merge and return 
    real_data = list(real_data)
    fake_data = list(fake_data) 
    return real_data + fake_data 

def fit(data, n_args=3, discount=.95, n_iters=500000, verbose=False, mini_batch=50):
    '''
    Fits a transfer-learned model on embedded data. Once fit, the 
    abstract model is combined with its lower parts (ie. convolutions) 
    and returned. Metrics are also returned to support later diagnostics. 
    '''
    ## Define model 
    model = Sequential() 
    model.add(ReLU(input_shape=(512,)))
    dense = Dense(n_args) 
    model.add(dense) 
    q_scores = model.output 
    action_ints = K.placeholder(shape=(None,), dtype='int32') 
    y = K.placeholder(shape=(None,), dtype='float32') 
    action_one_hots = K.one_hot(action_ints, n_args) 
    q_scores = K.sum(q_scores * action_one_hots, axis=1) 
    square_error = K.square(y - q_scores) 
    loss = K.mean(square_error)
    updates = RMSprop(lr=0.00025, epsilon=0.01).get_updates(loss, model.trainable_weights)
    train = K.function([model.input, action_ints, y], [loss], updates=updates) 
    ## Fit last layer of q-net on data 
    # build inputs from [(s_t, a_t, r_t, s_t+1, d_t)]_t data 
    states = np.array([tpl[0].flatten() for tpl in data]) 
    states = np.reshape(states, (-1, 512)) 
    next_states = np.array([tpl[3].flatten() for tpl in data])
    next_states = np.reshape(next_states, (-1, 512))
    action_ints = np.array([tpl[1] for tpl in data])
    rewards = np.array([tpl[2] for tpl in data])
    dones = np.array([int(tpl[4]) for tpl in data])
    losses = [] 
    stat_idx = random.sample(range(states.shape[0]), min(10, mini_batch))
    for itr in range(n_iters): 
        # iterate 
        idx = random.sample(range(states.shape[0]), mini_batch)
        y = rewards[idx] + (1-dones[idx]) * discount * np.amax(model.predict(next_states[idx,:]), axis=1) 
        l = train([states[idx,:], action_ints[idx], y]) 
        losses.append(l[0]) 
        if verbose:
            mean_q = np.mean(model.predict(states[stat_idx,:])) 
            print('%: ' + str(itr/float(n_iters)) + ', mean q: '+str(mean_q)) 
    ## Combine with lower transfer-learned layers
    weights = dense.get_weights() 
    FULL_RL_MODEL.model.layers[-1].set_weights(weights) 
    return model  

def metric_trials(sample_size = 1000, max_steps=10000): 
    '''
    Metric: Average score  
    '''
    t0 = time() 
    simulations = [] 
    for i in range(sample_size): 
        simulant = FULL_RL_MODEL.simulate(max_steps=max_steps) 
        simulations.append(simulant) 
        progress = str(100.*float(i+1)/float(sample_size))+'% complete in '+str(time()-t0)+' seconds\n' 
        # useful for spark-managed jobs 
        with open('/dat/progress.txt', 'a') as f: 
            f.write(progress) 
    return np.mean(simulations) 

def simple_eval_experiment(n_real=1000, n_fake=1000, metric_sample_size=1000, metric_max_steps=10000):
    '''
    Generates a single observation for a simple evaluation. 
    args:
     - `sample_size`: number of game transitions to sample. 
     - `probability_simulated`: how many samples are to be GAN-generated? 
    retruns:
     - `metric`: average score over iterated trials 
     - `losses`: model fitting losses, for diagnostics 
    '''
    data = simple_sample(n_real, n_fake) 
    _ = fit(data) 
    metric = metric_trials(metric_sample_size, metric_max_steps) 
    return metric 

def __sample_real_data(n):
    idx = random.sample(range(CVAE_DATA[1].shape[0]), n) 
    states = CVAE_DATA[0][idx,:] 
    labels = CVAE_DATA[1][idx] 
    return inverse_transfer_sample(states, list(labels)) 

def __sample_fake_data(n):
    '''
    Fake data is generated by a cGAN. Conditioned states are sampled 
    from a choice distribution. 
    '''
    # semi-stratified sampling over `(rewarded and dead)`. 
    # `action` assumed uniformly distributed. 
    labels = np.random.choice([0,1,2,3,4,5,6,7,8], p=[.01/3, .03/3, .96/3]*3, size=n) 
    fake_data_raw = CVAE_MODEL.generate(labels) 
    # data needs to be transformed into `(state_t, action_t, reward_t, state_t+1, dead_t)` 
    fake_data = inverse_transfer_sample(fake_data_raw, list(labels)) 
    return fake_data

if __name__ == '__main__':
    from time import sleep 
    while True:
        # This is for debugging. 
        # Don't deploy to a single node for actual work. 
        # It takes too long. 
        # Executing `simple_eval_experiment` with 1000 
        # samples is projected to take 3.6 hours (13s each). 
        # Many experiments need to be run. 
        # Parallelize instead. 
        print('Debug mode. Sleeping...') 
        sleep(100)  
    pass 















