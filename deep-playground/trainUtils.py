import numpy as np, pandas as pd, time, os
from keras.utils import Sequence
from keras.callbacks import Callback


class HistoryCallback(Callback):
    '''
    Optional functions to define can be:
       - on_(epoch|batch|train)_(begin|end)
    Expected arguments:
       - (self, (epoch|batch), logs={})
    '''
    def __init__(self, mod_name, log_file, video=None):
        self.mod_name = mod_name
        self.log_file = log_file
        self.video = video
    
    def on_train_begin(self, logs={}):
        self.epochs = []
        self.log_list = []
        self.times = []
        self.videos = []
        now = time.strftime('%A, %d %b %Y %H:%M:%S', time.gmtime(time.time() + 3600 * 2))
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\n\nStarting to train model {} on {}...'.format(self.mod_name, now))
        
    def on_epoch_begin(self, epoch, logs={}):
        self.init_time = time.time()
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\nStarting epoch {}, video {}...'.format(epoch, self.video))

    def on_epoch_end(self, epoch, logs={}):
        end_time = round(time.time() - self.init_time)
        self.epochs.append(epoch)
        self.log_list.append(logs.copy())
        self.times.append(end_time)
        self.videos.append(self.video)
        with open(self.log_file, 'a+') as f_log:
            f_log.write('\nIt took {}s'.format(end_time))
         
    def on_train_end(self, logs={}):
        hist = pd.DataFrame()
        hist['epoch'] = self.epochs
        if self.video is not None:
            hist['video'] = self.videos
        hist['duration [s]'] = self.times
        # Iterate on log keys (typically: loss, val_loss...)
        for col in self.log_list[0].keys():
            hist[col] = [log[col] for log in self.log_list]
        history_file = self.log_file.replace('logs.txt', 'hist.csv')
        if os.path.exists(history_file):
            prev_hist = pd.read_csv(history_file)
            hist = pd.concat([prev_hist, hist])
        hist.set_index('epoch').to_csv(history_file, index=True)
      
    
class DataGenerator(Sequence):
    '''
    Data generator for Keras (fit_generator). Based on:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    
    def __init__(self, list_IDs, files, X_reshape,
                 batch_size=32, shuffle=False, timestep=0):
        '''Initialization of the generator object'''
        # Note that list_IDs should be a np.array
        self.list_IDs = list_IDs
        self.n_samples = len(self.list_IDs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.timestep = timestep # <- for LSTMs
        self.ts_pointer = 0      # <- for LSTMs
        self.file_X = files['file_X']
        self.file_y = files['file_y']
        self.group_X = files['group_X']
        self.group_y = files['group_y']
        self.X_reshape = X_reshape if X_reshape[0] is not None else (batch_size, *X_reshape[1:])
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        if self.timestep == 0:
            # +1 if the number of samples is not % self.batch_size
            last_batch = 1 if self.n_samples % self.batch_size else 0
            return int(np.floor(self.n_samples / self.batch_size)) + last_batch
        #else: # <- for LSTMs
        steps = self.n_samples - self.timestep + 1
        return int(np.floor(steps / self.batch_size))

    def __getitem__(self, batch_n):
        '''Generate one batch of data'''
        # Generate data
        if self.timestep == 0:
            # Normal data generation, for NN (like cnn)
            # Generate indexes of the batch
            batch_slice = self.indexes[batch_n * self.batch_size:(batch_n+1) * self.batch_size]
            # Find list of IDs
            list_IDs_batch = self.list_IDs[batch_slice]
            X, y = self.__data_generation(list_IDs_batch)
            #X = X.reshape(self.X_reshape)
        else:
            # For RNNs/LSTMs/GRUs
            n_to_read = self.batch_size + self.timestep - 1
            # Read chunk
            X, y = self.__data_generation(np.arange(self.ts_pointer, self.ts_pointer + n_to_read))
            if self.batch_size > 1:
                # Reshape, copying slices of len = timestep
                X_tot = np.empty((self.X_reshape))
                for i in range(self.batch_size):
                    X_tot[i] = X[i:i + self.timestep].reshape(self.X_reshape[1:])
                X = X_tot
            else:
                X = X.reshape(self.X_reshape)
            y = y.iloc[-self.batch_size:]
            # Update the timestep pointer for the next batch
            self.ts_pointer += self.batch_size

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch, shuffling if necessary'''
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_batch):
        '''Generates data containing the required samples'''
        with pd.HDFStore(self.file_X, mode='r') as store_X, \
             pd.HDFStore(self.file_y, mode='r') as store_y:
            X = store_X.get_node(self.group_X)[list_IDs_batch, :]
            y = store_y.select(self.group_y).iloc[list_IDs_batch, 1]    
        
        return X, y