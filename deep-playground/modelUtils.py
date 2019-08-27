from keras import layers, models, optimizers
from keras.layers import TimeDistributed

class Models():
    '''
    Class that contains the hardcoded models.
    Also allows to load a pre-trained model from an HDF5 file.
    '''
    
    def __init__(self, loss='mse', metrics=['mae'],
                 optimizer=optimizers.RMSprop, lr=1e-4):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.lr = lr
        self.mod_list = ['cnn', 'cnn_v1', 'lstm', 'gru', 'gru_v1',
                         'cnn_lstm', 'cnn_lstm_v1', 'cnn_gru']

    def get_models_list(self):
        return self.mod_list
    
    def model(self, mod_name):
        if mod_name == 'cnn':
            return self.__cnn()
        elif mod_name == 'cnn_v1':
            return self.__cnn_v1()
        elif mod_name == 'lstm':
            return self.__lstm()
        elif mod_name == 'gru':
            return self.__gru()
        elif mod_name == 'gru_v1':
            return self.__gru_v1()
        elif mod_name == 'cnn_lstm':
            return self.__cnn_lstm()
        elif mod_name == 'cnn_lstm_v1':
            return self.__cnn_lstm_v1()
        elif mod_name == 'cnn_gru':
            return self.__cnn_gru()
        print('Model not found!')
        return None

    def load_model(self, model_path, do_compile=False):
        model = models.load_model(model_path)
        if do_compile:
            self.__compile_model(model)
        return model

    def __compile_model(self, model):
        model.compile(optimizer=self.optimizer(lr=self.lr), 
                      loss=self.loss, metrics=self.metrics)

    def __cnn(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='linear',
                                input_shape=(300, 300, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.LeakyReLU(alpha=0.3))
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
        # Compile model and return
        self.__compile_model(model)
        return model

    def __cnn_v1(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (5, 5), activation='linear',
                              input_shape=(300, 300, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Conv2D(64, (3, 3), activation='linear'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Conv2D(128, (4, 4), activation='linear'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __lstm(self):
        # np.prod((300, 300, 3)) == 270000
        model = models.Sequential()
        model.add(layers.recurrent.LSTM(units=1, stateful=True, batch_size=50,
                                        input_shape=(3, 270000),
                                        dropout=0.3, return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.recurrent.LSTM(units=128, stateful=True, dropout=0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(units=512, activation='linear'))
        model.add(layers.Dropout(0.4))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=64, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __gru(self):
        model = models.Sequential()
        model.add(layers.recurrent.GRU(units=1, stateful=True, batch_size=50,
                                        input_shape=(3, 270000),
                                        dropout=0.3, return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.recurrent.GRU(units=128, stateful=True, dropout=0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(units=512, activation='linear'))
        model.add(layers.Dropout(0.4))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=64, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __gru_v1(self):
        model = models.Sequential()
        model.add(layers.recurrent.GRU(units=1, stateful=True, batch_size=50,
                                        input_shape=(3, 270000),
                                        dropout=0.3, return_sequences=True))
        model.add(layers.recurrent.GRU(units=128, stateful=True, dropout=0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(units=64, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(units=1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __cnn_lstm(self):
        model = models.Sequential()
        model.add(TimeDistributed(layers.Conv2D(32, (3, 3), activation='linear'), batch_size=1,
                                  input_shape=(3, 300, 300, 3)))
        model.add(TimeDistributed(layers.MaxPooling2D((2, 2))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(TimeDistributed(layers.Conv2D(64, (3, 3), activation='linear')))
        model.add(TimeDistributed(layers.MaxPooling2D((4, 4))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Reshape(target_shape=(3, 36*36*64)))
        model.add(layers.recurrent.LSTM(units=3, stateful=True, dropout=0.3)) # return_sequences=True
        model.add(layers.Dense(units=128, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=512, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=64, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __cnn_lstm_v1(self):
        model = models.Sequential()
        model.add(TimeDistributed(layers.Conv2D(32, (3, 3), activation='linear'), batch_size=50,
                                  input_shape=(3, 300, 300, 3)))
        model.add(TimeDistributed(layers.MaxPooling2D((2, 2))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.BatchNormalization())
        model.add(TimeDistributed(layers.Conv2D(64, (3, 3), activation='linear')))
        model.add(TimeDistributed(layers.MaxPooling2D((4, 4))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(TimeDistributed(layers.Conv2D(16, (3, 3), activation='linear')))
        model.add(TimeDistributed(layers.MaxPooling2D((4, 4))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Reshape(target_shape=(3, 8*8*16)))
        model.add(layers.recurrent.LSTM(units=3, stateful=True,
                                        return_sequences=True, dropout=0.3))
        model.add(layers.recurrent.LSTM(units=128, dropout=0.3))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(units=512, activation='linear'))
        model.add(layers.Dropout(0.4))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=64, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __cnn_gru(self):
        model = models.Sequential()
        model.add(TimeDistributed(layers.Conv2D(32, (3, 3), activation='linear'), batch_size=50,
                                  input_shape=(3, 300, 300, 3)))
        model.add(TimeDistributed(layers.MaxPooling2D((2, 2))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.BatchNormalization())
        model.add(TimeDistributed(layers.Conv2D(64, (3, 3), activation='linear')))
        model.add(TimeDistributed(layers.MaxPooling2D((4, 4))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(TimeDistributed(layers.Conv2D(16, (3, 3), activation='linear')))
        model.add(TimeDistributed(layers.MaxPooling2D((4, 4))))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Reshape(target_shape=(3, 8*8*16)))
        model.add(layers.recurrent.GRU(units=3, stateful=True,
                                        return_sequences=True, dropout=0.3))
        model.add(layers.recurrent.GRU(units=128, dropout=0.3))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(units=512, activation='linear'))
        model.add(layers.Dropout(0.4))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=64, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        # Compile model and return
        self.__compile_model(model)
        return model

    # Optimizer
    def get_optimizer(self):
        return self.optimizer
    def set_optimizer(self, x):
        self.optimizer = x

    # Loss
    def get_loss(self):
        return self.loss
    def set_loss(self, x):
        self.loss = x

    # Metrics
    def get_metrics(self):
        return self.metrics
    def set_metrics(self, x):
        self.metrics = x

    # Learning rate
    def get_lr(self): 
        return self.lr 
    def set_lr(self, x): 
        self.lr = x
