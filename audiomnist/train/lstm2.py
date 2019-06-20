from ..io.read_dataset import load_audionet_dataset
from audiomnist.train.audionet import split
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from .splits import splits
import math

import os
import gc
import glob

size = 800

print("imported")

def train(dataset_path,checkpoint_path, logdir, batch_size, epochs):

    '''
    Load the data
    '''
    print("loading data")
    tf.enable_eager_execution()
    dataset = load_audionet_dataset(dataset_path)


    '''
    Compute max for standardization
    '''

    print("compute max")
    maxi = 0
    for e in dataset :
        l = e['data'].numpy().flatten()
        maxi = max(max(max(l),abs(min(l))),maxi)
    '''
    Split the dataset
    '''

    pre_train_dataset = dataset.filter(split('digit', 'train'))
    pre_test_dataset = dataset.filter(split('digit', 'test'))

    def fun_select_slice(i, size):
        def select_slice(record):
            return record['data'][i*size:(i+1)*size]
        return select_slice

    def split_dataset(dataset):
        sliced = [dataset.map(fun_select_slice(i, size)) for i in range(8000//size)]
        final_dataset = sliced[0]
        for d in sliced[1:]:
            final_dataset = final_dataset.concatenate(d)
            
        return final_dataset.map(lambda x: (x, x)) \
            .shuffle(18000, seed=42) \
            .batch(8000//size) \
            .repeat()

    train_dataset = split_dataset(pre_train_dataset)
    train_nb_samples = len(splits['digit']['train'][0])*500

    test_dataset = split_dataset(pre_test_dataset)
    test_nb_samples = len(splits['digit']['test'][0])*500

    '''
    Neural Net model
    '''
    print("building nn")

    timesteps = size
    input_dim = 1
    latent_dim = 100

    model = Sequential()
    model.add(LSTM(latent_dim, activation='relu', input_shape=(timesteps,1), stateful = True, batch_size=8000//size))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(latent_dim, activation='relu', return_sequences=True, stateful=True, batch_size=8000//size))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')

    print(model.summary())


    # adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # autoencoder.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    '''
    Callbacks
    '''
    if not os.path.isdir(logdir): os.mkdir(logdir)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                 batch_size=batch_size)

    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, "model.{epoch:02d}-{val_acc:.2f}"),
                                                            save_weights_only=True)

    gc_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch,_: gc.collect())

    model.fit(train_dataset, \
                epochs= epochs, \
                steps_per_epoch = math.ceil(train_nb_samples/batch_size), \
                shuffle=True, \
                validation_data=test_dataset, \
                validation_steps=math.ceil(test_nb_samples/batch_size), \
                callbacks = [tb_callback, checkpoint_callback], \
                verbose=1)
