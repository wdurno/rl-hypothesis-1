from __future__ import print_function, division

from google.cloud import storage
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
import pickle 

cgan_data_path = '/app/cgan-data.pkl'
cgan_statistics_path = '/app/cgan-statistics.pkl'
cgan_statistics_name = 'cgan-statistics.pkl'
cgan_model_path = '/app/cgan-model.h5'
cgan_model_name = 'cgan-model.h5'
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

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client.from_service_account_json('/app/service-account.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
    pass

if not os.path.isfile(cgan_data_path): 
    download_blob(bucket_name, 'cgan-data.pkl-backup', cgan_data_path) 
with open(cgan_data_path, 'rb') as f:
    data = pickle.load(f) 

class CGAN():
    def __init__(self):
        # Input shape
        self.img_shape = (1024,1) 
        self.num_classes = 3
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

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
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256)) 
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(Dense(256)) 
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(Dense(512)) 
        model.add(LeakyReLU(alpha=0.2)) 
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape)))
        model.add(ReLU(negative_slope=0.2, threshold=0.0)) 
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

        model.add(Dense(1024, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(126)) 
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

    def train(self, epochs, batch_size=128, sample_interval=50):
        
        statistics = {'d_loss': [], 'acc': [], 'g_loss': []} 

        # Load the dataset
        X_train, y_train = data

        # removing unfortunate transform 
        X_train += 3.5 

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
            statistics['d_loss'].append(d_loss[0]) 
            statistics['g_loss'].append(g_loss) 
            statistics['acc'].append(100*d_loss[1]) 

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # save and upload statistics 
                with open(cgan_statistics_path, 'wb') as f:
                    pickle.dump(statistics, f) 
                upload_blob(bucket_name, cgan_statistics_path, cgan_statistics_name) 
                # save and upload model 
                self.generator.save_weights(cgan_model_path) 
                upload_blob(bucket_name, cgan_model_path, cgan_model_name) 
                pass 


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=10000, sample_interval=200)
