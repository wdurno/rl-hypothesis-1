from gcp_api_wrapper import download_blob, upload_blob, shutdown
from time import sleep 
import os 
import pickle
import numpy as np
from random import choice 
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.utils import to_categorical
from scipy.stats import norm

EMBEDDING_DIM = int(os.environ['EMBEDDING_DIM']) 

cvae_data_path = '/dat/cvae-data.pkl'
cvae_data_name = 'cvae-data.pkl'
cvae_model_path = '/dat/cvae-model.h5'
cvae_model_name = 'cvae-model.h5'
cvae_simulated_data_path = '/dat/cvae-example-data.pkl'
cvae_simulated_data_name = 'cvae-example-data.pkl'
cvae_fit_stats_path = '/dat/cvae-fit-stats.pkl'
cvae_fit_stats_name = 'cvae-fit-stats.pkl'
cvae_embedding_sample_path = '/dat/cvae-embedding-sample.pkl'
cvae_embedding_sample_name = 'cvae-embedding-sample.pkl'

if not os.path.isfile(cvae_data_path):
    download_blob(cvae_data_name, cvae_data_path) 
with open(cvae_data_path, 'rb') as f:
    data = pickle.load(f)

class CVAE:
    '''
    Conditional Variational Autoencoder
    '''
    def __init__(self, data_dim, label_dim, latent_dim=100, n_hidden=EMBEDDING_DIM, model_path=None, batch_size=100000, n_epoch=300, kl_coef=.1):
        '''
        model_path: If `None`, then initialize an untrained model. Otherwise, load from the path. 
        '''
        ## store args 
        self.data_dim = data_dim 
        self.label_dim = label_dim
        self.latent_dim = latent_dim 
        self.n_hidden = n_hidden 
        self.model_path = model_path 
        self.batch_size = batch_size
        self.n_epoch = n_epoch 
        self.kl_coef = kl_coef 
        ## define model 
        self.__init_model()
        if model_path is not None: 
            self.cvae.load_weights(model_path)  
        pass
    
    def __init_model(self):
        '''
        Initializes model, does not load weights. 
        '''
        ## get args 
        data_dim = self.data_dim 
        label_dim = self.label_dim 
        latent_dim = self.latent_dim 
        n_hidden = self.n_hidden 
        batch_size = self.batch_size 
        kl_coef = self.kl_coef 
        ## encoder 
        x = Input(shape=(data_dim,)) 
        condition = Input(shape=(label_dim,))
        inputs = concatenate([x, condition]) 
        x_encoded = Dense(n_hidden, activation='relu')(inputs) 
        x_encoded = Dense(n_hidden//2, activation='relu')(x_encoded) 
        x_encoded = Dense(n_hidden//4, activation='relu')(x_encoded) 
        mu = Dense(latent_dim, activation='linear')(x_encoded) 
        log_var = Dense(latent_dim, activation='linear')(x_encoded) 
        ## latent sampler 
        def sampling(args): 
            mu, log_var = args 
            eps = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.) 
            return mu + K.exp(log_var/2.) * eps 
        ## sample 
        z = Lambda(sampling, output_shape=(latent_dim,))([mu, log_var]) 
        z_cond = concatenate([z, condition]) 
        ## decoder 
        z_decoder1 = Dense(n_hidden//4, activation='relu') 
        z_decoder2 = Dense(n_hidden//2, activation='relu') 
        z_decoder3 = Dense(n_hidden, activation='relu') 
        y_decoder = Dense(data_dim, activation='linear') 
        z_decoded = z_decoder1(z_cond) 
        z_decoded = z_decoder2(z_decoded) 
        z_decoded = z_decoder3(z_decoded) 
        y = y_decoder(z_decoded) 
        ## loss 
        reconstruction_loss = objectives.mean_squared_error(x, y) 
        kl_loss = .5 * K.mean(K.square(mu) + K.exp(log_var) - log_var - 1, axis=-1) 
        cvae_loss = reconstruction_loss + kl_coef * kl_loss 
        ## define full model 
        cvae = Model([x, condition], y) 
        cvae.add_loss(cvae_loss) 
        cvae.compile(optimizer='adam') 
        cvae.summary() 
        self.cvae = cvae 
        ## define encoder model 
        encoder = Model([x, condition], mu) 
        self.encoder = encoder 
        ## define decoder model 
        decoder_input = Input(shape=(latent_dim + label_dim,)) 
        _z_decoded = z_decoder1(decoder_input) 
        _z_decoded = z_decoder2(_z_decoded) 
        _z_decoded = z_decoder3(_z_decoded) 
        _y = y_decoder(_z_decoded) 
        generator = Model(decoder_input, _y) 
        generator.summary() 
        self.generator = generator 
        pass
    
    def __one_hot(self, arr): 
        arr = arr.astype(np.int32) 
        one_hots = np.zeros((arr.size, self.label_dim)) 
        one_hots[np.arange(arr.size), arr] = 1 
        return one_hots
    
    def fit(self):
        ## get args 
        batch_size = self.batch_size 
        n_epoch = self.n_epoch
        ## fit model 
        one_hots = self.__one_hot(data[1]) 
        return self.cvae.fit([data[0], one_hots],
                shuffle=True,
                epochs=n_epoch, 
                batch_size=batch_size,
                verbose=1)
    
    def generate(self, labels, apply_one_hot_transform=True):
        '''
        Generates data conditional on labels. 
        If labels are non-vector integers, use `apply_one_hot_transform=True`.
        '''
        labels = np.array(labels) 
        if apply_one_hot_transform:
            labels = self.__one_hot(labels) 
        z = np.random.normal(size=(labels.shape[0], self.latent_dim)) 
        return self.generator.predict(np.concatenate([z, labels], axis=1)) 
    
    def encode(self, n=10000):
        '''
        Randomly samples from `cvae-data.pkl` and encodes into the latent space. 
        '''
        if n < 0:
            n = data[0].shape[0] 
        # sample 
        idx = np.random.randint(data[0].shape[0], size=n) 
        x = np.take(data[0], idx, axis=0) 
        labels = np.take(data[1], idx, axis=0) 
        # transform to one-hots 
        labels = np.array(labels) 
        labels = self.__one_hot(labels) 
        # encode and return 
        return self.encoder.predict([x, labels]) 
        pass

    def save_model(self, path): 
        self.cvae.save_weights(path) 
    pass

if __name__ == '__main__':
    cvae = CVAE(data_dim=EMBEDDING_DIM*2, label_dim=9) 
    ## fit and save model 
    fit_stats = cvae.fit()
    fit_stats = [losses.mean() for losses in fit_stats.history['loss']] # get mean losses per epoch 
    cvae.save_model(cvae_model_path) 
    upload_blob(cvae_model_path, cvae_model_name) 
    ## save and upload loss stats 
    with open(cvae_fit_stats_path, 'wb') as f: 
        pickle.dump(fit_stats, f)
    upload_blob(cvae_fit_stats_path, cvae_fit_stats_name) 
    ## generate and save simulated data 
    cvae_simulated_data = cvae.generate([choice(range(9)) for _ in range(1000)]) 
    with open(cvae_simulated_data_path, 'wb') as f: 
        pickle.dump(cvae_simulated_data, f) 
    upload_blob(cvae_simulated_data_path, cvae_simulated_data_name) 
    ## save an embedding sample for diagnostics 
    cvae_embedding_sample = cvae.encode() 
    with open(cvae_embedding_sample_path, 'wb') as f:
        pickle.dump(cvae_embedding_sample, f) 
    upload_blob(cvae_embedding_sample_path, cvae_embedding_sample_name) 
    while True:
        shutdown()
        sleep(100) 
