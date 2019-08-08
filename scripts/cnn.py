import numpy as np
import pandas as pd
import psutil
from keras import layers, models, optimizers
from train_utils import DataGenerator

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

# Define parameters for the generator
params = {'batch_size': 2**5,
          'shuffle': False,
          'files' : {
            'file_X': '../data/clean/car_frames_norm_shuffled.h5',
            'file_y':  '../data/clean/car_distances_adjusted_shuffled.h5',
            'group_X': '/frames',
            'group_y': '/distances'
           }
         }

# Total number of samples
with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
    n_samples = store_y[params['files']['group_y']].shape[0]
validation_split = 0.1

# Indexes of the validation set
val_idx = np.random.choice(np.arange(n_samples), replace=False,
                           size=int(np.floor(n_samples * validation_split)))
# Indexes of the training set
train_idx = np.array(list(set(np.arange(n_samples)) - set(val_set)))

# Generators
train_generator = DataGenerator(train_idx, **params)
val_generator = DataGenerator(val_idx, **params)

# Define model and epochs
model = define_cnn()
compile_model(model)
n_epochs = 3

history = model.fit_generator(generator=train_generator,
                              validation_data=val_generator,
                              epochs=n_epochs, 
                              use_multiprocessing=True, workers=2) 