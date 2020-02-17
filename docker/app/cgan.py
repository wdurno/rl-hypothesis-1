## libraries 
from __future__ import print_function, division

from gcp_api_wrapper import download_blob, upload_blob, shutdown
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, ReLU 
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
# turn off plots 
import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys 
import pickle
from time import sleep 

## constants 
cgan_data_path = '/app/cgan-data.pkl'
cgan_statistics_path = '/app/cgan-statistics.pkl'
cgan_statistics_name = 'cgan-statistics.pkl'
cgan_model_path = '/app/cgan-model.h5'
cgan_model_name = 'cgan-model.h5'
cgan_discr_path = '/app/cgan-discr-model.h5' 
cgan_discr_name = 'cgan-discr-model.h5' 

if not os.path.isfile(cgan_data_path): 
    download_blob('cgan-data.pkl', cgan_data_path) 
with open(cgan_data_path, 'rb') as f:
    data = pickle.load(f) 

class CGAN():
    def __init__(self,
            load_discr_path=None,
            load_model_path=None
            ):
        '''
        Builds a cGAN.
        Args
         - load_discr_name: if not `None`, load discriminator weights. 
        '''
        # Input shape
        self.img_shape = (1024, 1) 
        self.num_classes = 9
        self.latent_dim = 100

        optimizer = Adam(0.003, 0.9)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        if load_discr_path is not None: 
            self.discriminator.load_weights(load_discr_path) 
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])
        
        # Build the generator
        self.generator = self.build_generator()
        if load_model_path is not None: 
            self.generator.load_weights(load_model_path)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512)) 
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(BatchNormalization(momentum=0.8)) 
        #model.add(Dense(512)) 
        #model.add(LeakyReLU(alpha=0.2)) 
        #model.add(BatchNormalization(momentum=0.8)) 
        model.add(Dense(np.prod(self.img_shape)))  
        #model.add(ReLU(negative_slope=0.2, threshold=0.0)) # data normally distributed 
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4)) 
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4)) 
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)
    
    @staticmethod 
    def __is_training_failure(statistics, min_steps=1000):
        '''
        Hueristics applied to ensure training has a chance of success. 
        Can detect a training failure. 
        Returns `True` if training has failed. 
        '''
        acc = statistics['acc'] 
        d_loss = statistics['d_loss'] 
        g_loss = statistics['g_loss'] 
        if len(acc) < min_steps:
            # too few observations to conclude failure 
            return False 
        acc_sub = acc[-min_steps:]
        if all([a < 10. for a in acc_sub]):
            # accuracy too low 
            return True 
        if all([a > 90. for a in acc_sub]): 
            # accuracy too high 
            return True 
        if all([a < 52. and a > 48. for a in acc_sub]): 
            # likely discriminator failure 
            return True 
        if len(acc) > 10000:
            if all([a < 20. for a in acc_sub]): 
                return True 
            if all([a > 80. for a in acc_sub]): 
                return True 
        # no strong evidence of failure 
        return False 

    def train(self, epochs, batch_size=128, sample_interval=50):
        '''
        Returns `True` on success. 
        Returns `False` on likely failure. 
        '''
        
        statistics = {'d_loss': [], 'acc': [], 'g_loss': []} 

        # Load the dataset
        X_train, y_train = data

        # Configure input
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # similar to images 
        X_train = X_train.astype(np.float32) 
        #X_train = np.expand_dims(X_train, axis=3) # seems specific to mnist 
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            sys.stdout.flush() 
            statistics['d_loss'].append(d_loss[0]) 
            statistics['g_loss'].append(g_loss) 
            statistics['acc'].append(100*d_loss[1]) 

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # check for failure 
                if CGAN.__is_training_failure(statistics, min_steps=2000):
                    return False 
                # save and upload statistics 
                with open(cgan_statistics_path, 'wb') as f:
                    pickle.dump(statistics, f) 
                upload_blob(cgan_statistics_path, cgan_statistics_name) 
                # save and upload generator  
                self.generator.save_weights(cgan_model_path) 
                upload_blob(cgan_model_path, cgan_model_name) 
                # save and upload discriminator 
                self.discriminator.save_weights(cgan_discr_path) 
                upload_blob(cgan_discr_path, cgan_discr_name) 
                pass
        # training complete 
        return True 

if __name__ == '__main__':
    continue_training = True
    attempt_num = 0 
    while continue_training: 
        print('TRAINING ATTEMPT ' + str(attempt_num))
        cgan = CGAN()
        continue_training = not cgan.train(epochs=25000, batch_size=100, sample_interval=200) 
        attempt_num += 1 
    while True: 
        shutdown()
        sleep(100) 
