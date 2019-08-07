import numpy as np
import pandas as pd
import psutil
from keras import layers, models, optimizers
#from keras.layers.wrappers import TimeDistributed


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

def define_cnn1():
    model2 = models.Sequential()
    model2.add(layers.Conv2D(64,(8,8), activation='linear', padding='same',
                                   input_shape=(300, 300, 3)))
    #Suggestion: add more than 8 filters! Try for example 24, 32, 64...
    #model2.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model2.add(layers.Dropout(rate=0.3))

    model2.add(layers.Conv2D(64,(8,8),padding='same', activation='linear'))
    model2.add(layers.MaxPooling2D(pool_size=(2,2)))
    model2.add(layers.LeakyReLU(alpha=0.3))
    model2.add(layers.Dropout(rate=0.3))

    model2.add(layers.Conv2D(16,(4,4),padding='same', activation='linear'))
    model2.add(layers.MaxPooling2D(pool_size=(2,2)))
    model2.add(layers.LeakyReLU(alpha=0.3))
    model2.add(layers.Dropout(rate=0.2))

    model2.add(layers.Flatten())
    #Sugestion: add a Dense layer, with relu activation here! Try 50, 100, 200 nodes...
    model2.add(layers.Dense(64, activation='linear'))
    model2.add(layers.LeakyReLU(alpha=0.3))
    model2.add(layers.Dropout(rate=0.2))
    model2.add(layers.Dense(512, activation='linear'))
    model2.add(layers.LeakyReLU(alpha=0.3))
    model2.add(layers.Dropout(rate=0.2))
    model2.add(layers.Dense(32, activation='linear'))
    model2.add(layers.LeakyReLU(alpha=0.3))
    model2.add(layers.Dropout(rate=0.1))
    model2.add(layers.Dense(1, activation='linear'))
    model2.add(layers.LeakyReLU(alpha=0.3))
    
    return model2

def compile_model(model):
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
                  loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
    return

#def define_lstm():
#    model = models.Sequential()
#    model.add(TimeDistributed(layers.Conv2D(32, (3, 3), activation='linear'),
#                            input_shape=(10, 300, 300, 3)))
#    model.add(TimeDistributed(layers.LeakyReLU(alpha=0.3)))
#    model.add(TimeDistributed(layers.MaxPooling2D((2, 2))))
#    model.add(TimeDistributed(layers.Conv2D(64, (3, 3), activation='linear')))
#    model.add(TimeDistributed(layers.MaxPooling2D((2, 2))))
#    model.add(TimeDistributed(layers.LeakyReLU(alpha=0.3)))
#    model.add(TimeDistributed(layers.Conv2D(128, (3, 3), activation='linear')))
#    model.add(TimeDistributed(layers.MaxPooling2D((2, 2))))
#    model.add(TimeDistributed(layers.LeakyReLU(alpha=0.3)))
#    model.add(TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')))
#    model.add(TimeDistributed(layers.MaxPooling2D((4, 4))))
#    model.add(TimeDistributed(layers.Flatten()))
#    model.add(layers.recurrent.LSTM(1024))
#    model.add(layers.Dense(512, activation='linear'))
#    model.add(layers.LeakyReLU(alpha=0.3))
#    model.add(layers.Dense(1, activation='linear'))
#    model.add(layers.LeakyReLU(alpha=0.3))
#    return model

    
#models = [define_cnn(), define_cnn1()]
from keras.models import load_model
models = [load_model('../data/models/cnn_v2.h5')]

n_epochs = 3
batch_size = 2**5
max_frames = 2**15
validation_split = 0.2
frames_path = '../data/clean/car_frames.h5'
dist_path = '../data/clean/car_distances_adjusted.h5'
store_f = pd.HDFStore(frames_path, mode='r')
store_d = pd.HDFStore(dist_path, mode='r')
n_videos = len(store_d)

for mod_name, model in zip(['cnn_v4'], models):
    compile_model(model)
    print('\nMODEL', mod_name, '\n')
    idx_video = [vid for vid in store_d] * n_epochs
    idx_epoch = [elem for sublist in [[i] * n_videos for i in range(n_epochs)] for elem in sublist]
    hist = pd.DataFrame(data=np.empty(shape=(n_videos * n_epochs, 4)), index=[idx_epoch, idx_video],
                        columns=['loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error'])
    hist[:] = np.nan
    hist.index.names = ['epoch', 'video']    
    for epoch in range(2, n_epochs):
        print('Starting epoch {}/{}...'.format(epoch + 1, n_epochs))
        for idx, video in enumerate(store_d):
            # Leave the Tesla-Mercedes video as a test set
            if video in ['/V0420043']:
                print('{}. Skipping {} for testing'.format(idx, video))
                continue
            print('{}. Video {}'.format(idx, video))
            frames = store_f.get_node(video)[:max_frames] / 255.0
            df = store_d.select(video).iloc[:max_frames, 1]
            # Shuffle indexes
            shuffled = np.arange(df.iloc[:max_frames].shape[0])
            if 'lstm' not in mod_name:
                shuffled = np.random.permutation(shuffled)
            history = model.fit(x=frames[shuffled, :], y=df.iloc[shuffled],
                                validation_split=validation_split, batch_size=batch_size,
                                initial_epoch=epoch, epochs=epoch+1)
            id_hist = n_videos * (epoch) + idx
            hist.iloc[id_hist, :] = (history.history['loss'][0], history.history['val_loss'][0],
                                     history.history['mean_absolute_error'][0], history.history['val_mean_absolute_error'][0])
            print('Memory: used {}GB, available {}GB'.format(psutil.virtual_memory().used >> 30,
                                                                psutil.virtual_memory().available >> 30))
    model.save('../data/models/{}.h5'.format(mod_name))
    hist.to_csv('../data/models/{}_hist.csv'.format(mod_name), index=True)
            
store_f.close()
store_d.close()
