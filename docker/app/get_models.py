## libraries
from gcp_api_wrapper import download_blob 
import sys 
import os 

## constants
# TODO duplications exist--refactor needed 
CVAE_DATA_PATH='/dat/cvae-data.pkl'
CVAE_DATA_BLOB_NAME='cvae-data.pkl'
CVAE_MODEL_PATH='/dat/cvae-model.h5'
CVAE_MODEL_BLOB_NAME='cvae-model.h5'
FULL_RL_MODEL_PATH='/dat/breakout_dqn.h5'
FULL_RL_MODEL_BLOB_NAME='rl-full.h5'
TRANSFER_MODEL_PATH = '/dat/transfer-model.h5'
TRANSFER_MODEL_BLOB_NAME = 'rl-full.h5-backup' 
FULL_SIM_DATA_PATH = '/dat/data.pkl'
FULL_SIM_DATA_BLOB_NAME = 'memory.pkl-backup' 

# ensure files are downloaded
if True: #not os.path.isfile(CVAE_DATA_PATH):
    download_blob(CVAE_DATA_BLOB_NAME, CVAE_DATA_PATH)
if True: #not os.path.isfile(CVAE_MODEL_PATH):
    download_blob(CVAE_MODEL_BLOB_NAME, CVAE_MODEL_PATH)
if True: #not os.path.isfile(FULL_RL_MODEL_PATH):
    download_blob(FULL_RL_MODEL_BLOB_NAME, FULL_RL_MODEL_PATH)
if True: # not os.path.isfile(TRANSFER_MODEL_PATH):
    download_blob(TRANSFER_MODEL_BLOB_NAME, TRANSFER_MODEL_PATH)
if True: # not os.path.isfile(FULL_SIM_DATA_PATH):
    download_blob(FULL_SIM_DATA_BLOB_NAME, FULL_SIM_DATA_PATH)

print('Model files downloaded') 
sys.stdout.flush()
