## libraries
from gcp_api_wrapper import download_blob 
import sys 
import os 

## constants
# TODO duplications exist--refactor needed 
CGAN_DATA_PATH='/dat/cgan-data.pkl'
CGAN_DATA_BLOB_NAME='cgan-data.pkl'
CGAN_MODEL_PATH='/dat/cgan-model.h5'
CGAN_MODEL_BLOB_NAME='cgan-model.h5'
FULL_RL_MODEL_PATH='/dat/breakout_dqn.h5'
FULL_RL_MODEL_BLOB_NAME='rl-full.h5'
TRANSFER_MODEL_PATH = '/dat/transfer-model.h5'
TRANSFER_MODEL_BLOB_NAME = 'rl-full.h5-backup' 
FULL_SIM_DATA_PATH = '/dat/data.pkl'
FULL_SIM_DATA_BLOB_NAME = 'memory.pkl-backup' 

# ensure files are downloaded
if True: #not os.path.isfile(CGAN_DATA_PATH):
    download_blob(CGAN_DATA_BLOB_NAME, CGAN_DATA_PATH)
if True: #not os.path.isfile(CGAN_MODEL_PATH):
    download_blob(CGAN_MODEL_BLOB_NAME, CGAN_MODEL_PATH)
if True: #not os.path.isfile(FULL_RL_MODEL_PATH):
    download_blob(FULL_RL_MODEL_BLOB_NAME, FULL_RL_MODEL_PATH)
if True: # not os.path.isfile(TRANSFER_MODEL_PATH):
    download_blob(TRANSFER_MODEL_BLOB_NAME, TRANSFER_MODEL_PATH)
if True: # not os.path.isfile(FULL_SIM_DATA_PATH):
    download_blob(FULL_SIM_DATA_BLOB_NAME, FULL_SIM_DATA_PATH)

print('Model files downloaded') 
sys.stdout.flush()
