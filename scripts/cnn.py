import numpy as np
import pandas as pd
import psutil
from keras import layers, models, optimizers


def define_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='linear',
                            input_shape=(300, 300, 3)))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='linear'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='linear'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='linear'))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.Dense(1, activation='linear'))
    model.add(layers.LeakyReLU(alpha=0.3))
    return model

def compile_model(model):
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
    return

def train_generator(frames_path, dist_path, batch_size, train_split):
    while True:
        with pd.HDFStore(frames_path, mode='r') as store_f, \
             pd.HDFStore(dist_path, mode='r') as store_d:
            n_batches = store_d.select('/distances').shape[0] // batch_size + 1
            for batch in range(0, int(np.floor(train_split * n_batches))):
                slc = slice(batch * batch_size, (batch+1) * batch_size)
                yield (store_f.get_node('/frames')[slc], 
                       store_d.select('/distances').iloc[slc, 1])

def val_generator(frames_path, dist_path, batch_size, validation_split):
    while True:
        with pd.HDFStore(frames_path, mode='r') as store_f, \
             pd.HDFStore(dist_path, mode='r') as store_d:
            n_batches = store_d.select('/distances').shape[0] // batch_size + 1
            for batch in range(int(np.ceil((1 - validation_split) * n_batches)), n_batches):
                slc = slice(batch * batch_size, (batch+1) * batch_size)
                yield (store_f.get_node('/frames')[slc], 
                       store_d.select('/distances').iloc[slc, 1])

models = [define_cnn()]

n_epochs = 2
batch_size = 2**5
validation_split = 0.1
frames_path = '../data/clean/car_frames_norm_shuffled.h5'
dist_path = '../data/clean/car_distances_adjusted_shuffled.h5'

for mod_name, model in zip(['cnn_v1_shuffled'], models):
    compile_model(model)
    print('\nMODEL', mod_name, '\n')
    history = model.fit_generator(generator=train_generator(frames_path, dist_path, batch_size, 1 - validation_split),
                                  validation_data=val_generator(frames_path, dist_path, batch_size, validation_split),
                                  workers=0, epochs=n_epochs, # initial_epoch=epoch
                                  steps_per_epoch=3587, validation_steps=398)
    print('Memory: used {}GB, available {}GB'.format(psutil.virtual_memory().used >> 30,
                                                     psutil.virtual_memory().available >> 30))
    model.save('../data/models/{}.h5'.format(mod_name))
    hist.to_csv('../data/models/{}_hist.csv'.format(mod_name), index=True)
