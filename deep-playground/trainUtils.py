import numpy as np, pandas as pd, time
from keras.utils import Sequence
from keras.callbacks import Callback


class HistoryCallback(Callback):
    '''
    Optional functions to define can be:
       - on_(epoch|batch|train)_(begin|end)
    Expected arguments:
       - (self, (epoch|batch), logs={})
    '''
    def __init__(self, mod_name):
        self.mod_name = mod_name
    
    def on_train_begin(self, logs={}):
        self.epochs = []
        self.logs = []
        self.times = []
        now = time.strftime('%A, %d %b %Y %H:%M:%S', time.gmtime(time.time() + 3600 * 2))
        with open('../data/models/a_logs.txt', 'a+') as f_log:
            f_log.write('\n\nStarting to train model {} on {}...'.format(self.mod_name, now))
        
    def on_epoch_begin(self, epoch, logs={}):
        self.init_time = time.time()
        with open('../data/models/a_logs.txt', 'a+') as f_log:
            f_log.write('\nStarting epoch {}...'.format(epoch))

    def on_epoch_end(self, epoch, logs={}):
        end_time = round(time.time() - self.init_time)
        self.epochs.append(epoch)
        self.logs.append(logs)
        self.times.append(end_time)
        with open('../data/models/a_logs.txt', 'a+') as f_log:
            f_log.write('\nIt took {}s'.format(end_time))
         
    def on_train_end(self, logs={}):
        hist = pd.DataFrame()
        hist['epoch'] = self.epochs
        hist['duration [s]'] = self.times
        # Iterate on log keys (typically: loss, val_loss...)
        for col in self.logs[0].keys():
            hist[col] = [log[col] for log in self.logs]
        hist.set_index('epoch').to_csv('../data/models/{}_hist.csv'.format(self.mod_name), index=True)
      
    
class DataGenerator(Sequence):
    '''
    Data generator for Keras (fit_generator). Based on:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''
    
    def __init__(self, list_IDs, files, batch_size=32, shuffle=False):
        '''Initialization of the generator object'''
        # Note that list_IDs should be a np.array
        self.list_IDs = list_IDs
        self.n_samples = len(self.list_IDs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_X = files['file_X']
        self.file_y = files['file_y']
        self.group_X = files['group_X']
        self.group_y = files['group_y']
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        # +1 if the number of samples is not % self.batch_size
        last_batch = 1 if self.n_samples % self.batch_size else 0
        return int(np.floor(self.n_samples / self.batch_size)) + last_batch

    def __getitem__(self, batch_n):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        batch_slice = self.indexes[batch_n * self.batch_size:(batch_n+1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = self.list_IDs[batch_slice]
            
        # Generate data
        X, y = self.__data_generation(list_IDs_batch)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch, shuffling if necessary'''
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_batch):
        '''Generates data containing batch_size samples'''
        with pd.HDFStore(self.file_X, mode='r') as store_X, \
             pd.HDFStore(self.file_y, mode='r') as store_y:
            X = store_X.get_node(self.group_X)[list_IDs_batch, :]
            y = store_y.select(self.group_y).iloc[list_IDs_batch, 1]    
        
        return X, y