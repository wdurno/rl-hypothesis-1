from google.cloud import storage
import os
import random 
import pickle 
from keras.models import Sequential, Model 
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D

model_path = '/app/transfer-model.h5' 
data_path = '/app/data.pkl'
bucket_name = os.environ['BUCKET_NAME'] 

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.from_service_account_json('/app/service-account.json') 

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
    pass

# download model only if needed 
if not os.path.isfile(model_path): 
    download_blob(bucket_name, 'rl-full.h5-backup', model_path) 
    pass

# download data only if needed 
if not os.path.isfile(data_path): 
    download_blob(bucket_name, 'memory.pkl-backup', data_path) 
    pass

with open(data_path , 'rb') as f: 
    data = pickle.load(f)
    pass 

def build_model(state_size, action_size):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_size))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size))
    model.summary()
    return model

full_model = build_model((84, 84, 4), 3) 
full_model.load_weights(model_path) 

rl_1_dense = Model(inputs=full_model.inputs, outputs=full_model.layers[4].output)
rl_convs = Model(inputs=full_model.inputs, outputs=full_model.layers[3].output)

def _transform_rl_observation(rl_observation, model=rl_1_dense):
    # state, action, reward, next_state, dead
    rl_observation = list(rl_observation) 
    rl_observation[0] = model.predict(rl_observation[0]) # state 
    rl_observation[3] = model.predict(rl_observation[3]) # next_state 
    return rl_observation 

def transfer_sample(n=10000, model=rl_1_dense): 
    '''
    Generates a random list of n RL observations. 
    Transfer sampling is applied. 
    '''
    sample = random.sample(data, n) 
    return list(map(_transform_rl_observation, sample))








