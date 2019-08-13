from keras import layers, models, optimizers

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
        self.mod_list = ['cnn', 'lstm', 'cnn_lstm']

    def get_models_list(self):
        return self.mod_list
    
    def model(self, mod_name):
        if mod_name == 'cnn':
            return self.__cnn()
        elif mod_name == 'lstm':
            return self.__lstm()
        elif mod_name == 'cnn_lstm':
            return self.__cnn_lstm()
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
        # Compile model and return
        self.__compile_model(model)
        return model

    def __lstm(self):
        # np.prod((300, 300, 3)) == 270000
        model = models.Sequential()
        model.add(layers.recurrent.LSTM(units=1, stateful=True, batch_size=1,
                                        input_shape=(10, 270000),
                                        dropout=0.3, return_sequences=True))
        model.add(layers.recurrent.LSTM(units=128, stateful=True, dropout=0.3))
        model.add(layers.Dense(units=512, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=64, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        model.add(layers.Dense(units=1, activation='linear'))
        model.add(layers.LeakyReLU(alpha=0.3))
        # Compile model and return
        self.__compile_model(model)
        return model

    def __cnn_lstm(self):
        model = models.Sequential()
     #   model.add(layers.Conv2D(32, (3, 3), activation='linear', batch_size=1,
     #                      input_shape=(10*300, 300, 3)))
     #   model.add(layers.LeakyReLU(alpha=0.3))
     #   model.add(layers.MaxPooling2D((2, 2)))
     #   model.add(layers.Conv2D(64, (3, 3), activation='linear'))
     #   model.add(layers.MaxPooling2D((4, 4)))
     #   model.add(layers.LeakyReLU(alpha=0.3))
     #   model.add(layers.Reshape(target_shape=(36, 36*64)))
     #   ##modl1.add(layers.Reshape(target_shape=(22498, 64)))
     #   model.add(layers.recurrent.LSTM(units=1, stateful=True,
     #                                   dropout=0.3, return_sequences=True))
     #   model.add(layers.Reshape(target_shape=(36,1)))
     #   model.add(layers.recurrent.LSTM(units=128, stateful=True, dropout=0.3))
     #   model.add(layers.Dense(units=512, activation='linear'))
     #   model.add(layers.LeakyReLU(alpha=0.3))
     #   model.add(layers.Dense(units=64, activation='linear'))
     #   model.add(layers.LeakyReLU(alpha=0.3))
     #   model.add(layers.Dense(units=1, activation='linear'))
     #   model.add(layers.LeakyReLU(alpha=0.3))
     #   # Compile model and return
     #   self.__compile_model(model)
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
