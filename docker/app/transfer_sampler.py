from gcp_api_wrapper import download_blob, upload_blob, shutdown
import os
import random 
import pickle
import numpy as np
from time import sleep 
from keras.models import Sequential, Model 
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import ReLU

model_path = '/app/transfer-model.h5' 
data_path = '/app/data.pkl'
cgan_data_path = '/app/cgan-data.pkl'

# download model only if needed 
if not os.path.isfile(model_path): 
    download_blob('rl-full.h5-backup', model_path) 
    pass

# download data only if needed 
if not os.path.isfile(data_path): 
    download_blob('memory.pkl-backup', data_path) 
    pass

with open(data_path , 'rb') as f: 
    data = pickle.load(f)
    pass 

def build_model(state_size, action_size):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Flatten())
    model.add(ReLU())
    model.add(Dense(512))
    model.add(ReLU())
    model.add(Dense(action_size))
    model.summary()
    return model

full_model = build_model((84, 84, 4), 3) 
full_model.load_weights(model_path) 

rl_1_dense = Model(inputs=full_model.inputs, outputs=full_model.layers[5].output)
rl_convs = Model(inputs=full_model.inputs, outputs=full_model.layers[3].output)

def _transfer_transform_rl_observation(rl_observation, model=rl_1_dense):
    '''
    Applies a transfer sampling transform. 
    '''
    # state, action, reward, next_state, dead
    rl_observation = list(rl_observation) 
    rl_observation[0] = model.predict(rl_observation[0]) # state 
    rl_observation[3] = model.predict(rl_observation[3]) # next_state 
    return rl_observation 

def _map_reward_dead_to_int(rl_observation): 
    rewarded = rl_observation[2] > .5 
    dead = rl_observation[4]
    action = rl_observation[1] 
    if (not rewarded) and dead and action == 0: 
        return 0
    elif rewarded and (not dead) and action == 0:
        return 1
    elif (not rewarded) and (not dead) and action == 0:
        return 2
    elif (not rewarded) and dead and action == 1:
        return 3
    elif rewarded and (not dead) and action == 1:
        return 4
    elif (not rewarded) and (not dead) and action == 1:
        return 5
    elif (not rewarded) and dead and action == 2:
        return 6
    elif rewarded and (not dead) and action == 2:
        return 7
    elif (not rewarded) and (not dead) and action == 2:
        return 8
    # this should not occur 
    return None  

def _map_int_to_action_reward_dead(state_int):
    "returns an (action, reward, dead) tuple"
    if state_int == 0:
        return 0, 0., True
    elif state_int == 1:
        return 0, 1., False
    elif state_int == 2:
        return 0, 0., False
    elif state_int == 3:
        return 1, 0., True
    elif state_int == 4:
        return 1, 1., False
    elif state_int == 5:
        return 1, 0., False
    elif state_int == 6:
        return 2, 0., True
    elif state_int == 7:
        return 2, 1., False
    elif state_int == 8:
        return 2, 0., False
    # state_int == 3 
    # this should not occur 
    return None  

def _map_transfers_to_array(transfer_transformed_rl_observation):
    transfer_state = transfer_transformed_rl_observation[0] 
    transfer_next_state = transfer_transformed_rl_observation[3] 
    # clipped deltas are approximately normal 
    diff = np.clip(transfer_next_state - transfer_state, -150., 150.) 
    return np.concatenate([np.clip(transfer_state, -400., 400), diff], axis=1) 

def _map_array_to_transfers(transfer_array, split_point=512): 
    "returns state, next_state"
    before = transfer_array[:, split_point:] 
    diff = transfer_array[:, :split_point]
    after = before + diff
    return before, after 

def transfer_sample(n=10000, model=rl_1_dense): 
    '''
    Generates a random list of n RL observations. 
    Transfer sampling is applied. 
    '''
    if n > 0:
        sample = random.sample(data, n) 
    else:
        sample = data
    return list(map(_transfer_transform_rl_observation, sample))

def inverse_transfer_sample():
    '''
    Extracts q-learning data in the form of `(s_t, r_t, s_t+1, d_t)`.
    No actions are returned because `s_t` are embeddings, encoding all action information. This won't work... 
    '''
    pass

def cgan_sample(n=10000, model=rl_1_dense):
    '''
    Generates a random sample of transformed RL observations.
    Transfer learning transform is applied.
    Data is formatted for cGAN fitting.
    '''
    tr = transfer_sample(n, model)
    labels = np.array(list(map(_map_reward_dead_to_int, tr)))
    states = np.concatenate(list(map(_map_transfers_to_array, tr)))
    return states, labels

def transform_all_and_upload(model=rl_1_dense): 
    sample_tuple = cgan_sample(n=-1, model=model) 
    with open(cgan_data_path, 'wb') as f: 
        pickle.dump(sample_tuple, f) 
    upload_blob(cgan_data_path, 'cgan-data.pkl') 
    pass 

if __name__ == '__main__':
    transform_all_and_upload()
    while True:
        shutdown() 
        sleep(100) 

