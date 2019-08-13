import numpy as np
import pandas as pd
import psutil, os, keras, time
from importlib import reload
from trainUtils import HistoryCallback, DataGenerator
import modelUtils
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#################################################
#                  TRAINING                     #
#################################################

def train_sequentially(model, mod_name, n_epochs):
    '''For the case of LSTMs.'''
    # Define parameters for the generator
    params = {'batch_size': 1,
              'shuffle': False,
              'X_reshape': model.layers[0].input_shape,
              'timestep': 10, # for LSTMs
              'files' : {
                'file_X': '../data/clean/car_frames_norm.h5',
                'file_y': '../data/clean/car_distances_adjusted.h5',
                'group_X': '',
                'group_y': ''
               }
             }
    # Extract video names
    with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
        videos = [video for video in store_y]
    for epoch in range(n_epochs):
        for idx, video in enumerate(videos):
            # Leave the Tesla-Mercedes video as a test set
            if video in ['/V0420043']:
                continue
            # Update params with the new video name
            params['files']['group_X'], params['files']['group_y'] = video, video
            # Total number of samples
            with pd.HDFStore('../data/clean/car_distances_adjusted.h5', mode='r') as store_y:
                n_samples = store_y[video].shape[0]
            train_split = 0.9
            # Indexes for train and validation
            train_idx = np.arange(n_samples)[0:int(n_samples * train_split)]
            val_idx = np.arange(n_samples)[int(n_samples * train_split):n_samples]

            # Generators
            train_generator = DataGenerator(train_idx, **params)
            val_generator = DataGenerator(val_idx, **params)

            model.fit_generator(generator=train_generator,
                                validation_data=val_generator, shuffle=False,
                                initial_epoch=epoch, epochs=epoch+1,
                                callbacks=[HistoryCallback(mod_name=mod_name, video=video,
                                                           log_file='logs_' + mod_name)],
                                use_multiprocessing=True, workers=2)
            model.reset_states()
    return model

def train_shuffled(model, mod_name, n_epochs):
    '''For the case of other NN (like CNNs).'''
    # Define parameters for the generator
    params = {'batch_size': 2**5,
              'shuffle': False,
              'X_reshape': model.layers[0].input_shape,
              'files' : {
                'file_X': '../data/clean/car_frames_norm_shuffled.h5',
                'file_y': '../data/clean/car_distances_adjusted_shuffled.h5',
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
    train_idx = np.array(list(set(np.arange(n_samples)) - set(val_idx)))

    # Generators
    train_generator = DataGenerator(train_idx, **params)
    val_generator = DataGenerator(val_idx, **params)

    model.fit_generator(generator=train_generator,
                        validation_data=val_generator, #shuffle=False,
                        epochs=n_epochs, callbacks=[HistoryCallback(mod_name)],
                        use_multiprocessing=True, workers=2)
    return model

def train_models(mod_list, mod_names, epochs_list):
    for idx, model in enumerate(mod_list):
        mod_name, n_epochs = mod_names[idx], epochs_list[idx]
        if not os.path.exists('../data/models/{}/'.format(mod_name)):
            os.mkdir('../data/models/{}/'.format(mod_name))
        print('\n\n\t{}. Starting to train model {}...\n'.format(idx, mod_name))
        if any(sb_str in mod_name for sb_str in ['rnn', 'lstm', 'gru']):
            model = train_sequentially(model, mod_name, n_epochs)
        else:
            model = train_shuffled(model, mod_name, n_epochs)
        model.save('../data/models/{}/{}.h5'.format(mod_name, mod_name))


def prepare_training(models):
    mod_list, mod_names, epochs_list = [], [], []
    while True:
        print('\n\t1. Create new model.')
        print('\t2. Load existing model.')
        print('\t3. Check models that will be trained.')
        print('\t4. Ready for training.')
        print('\t5. Back to menu.')
        option = int(input('\n\tPlease enter your option: '))
        if option == 1:
            print('\n\tAvailable models are: {}'.format(models.get_models_list()))
            mod_abbr = input('\n\tModel to create: ')
            if mod_abbr in models.get_models_list():
                model = models.model(mod_abbr)
            else: print('\n\tWrong model!'); continue
        elif option == 2: model, _ = load_model(models)
        elif option == 3: print('\n\t{}'.format(list(zip(mod_names, epochs_list)))); continue
        elif option == 4: train_models(mod_list, mod_names, epochs_list); break
        elif option == 5: break
        else:
            print('\n\tInvalid option!')
            input('\n\tPress Enter to continue...')
            continue
        mod_list.append(model)
        mod_names.append(input('\n\tPlease enter a model name: '))
        epochs_list.append(int(input('\n\tNumber of epochs: ')))
        if input('\n\tSee model summary? [y/n]: ') == 'y':
            model.summary()
        print('\n\tModel added to queue.')


#################################################
#                  TESTING                      #
#################################################

def loss_plot(model, mod_name):
    # Load history file
    hist = pd.read_csv('../data/models/{}/{}_hist.csv'.format(mod_name, mod_name))
    days = hist['duration [s]'].sum() // (24 * 3600)
    print('\n\tTotal training time for model {} was {} days and {}'.format(mod_name, days,
        time.strftime('%H:%M:%S', time.gmtime(hist['duration [s]'].sum()))))
    # Plot loss and metrics
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(15, 10)
    axs[0].plot(hist['val_mean_absolute_error'])
    axs[0].plot(hist['mean_absolute_error'])
    axs[0].legend(['val_mean_absolute_error', 'mean_absolute_error'], fontsize=12)
    axs[0].set_title('Mean Absolute Error', fontsize=15)
    axs[0].set_yscale('log')
    axs[0].grid()
    axs[1].plot(hist['val_loss'])
    axs[1].plot(hist['loss'])
    axs[1].legend(['val_loss', 'loss'], fontsize=12)
    axs[1].set_title('Mean Squared Error', fontsize=15)
    axs[1].set_yscale('log')
    axs[1].grid()
    # Save resulting figure
    fig.savefig('../data/models/{}/{}_loss.png'.format(mod_name, mod_name))
    print('\n\tLoss plots saved in ../data/models/{}/{}_loss.png'.format(mod_name, mod_name))
    return

def scatter_plot(y_true, y_pred, mod_name, unseen_shuffled=''):
    max_val = max(max(y_true), max(y_pred)) + 5
    length = len(y_true)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    ax.plot([0, max_val], [0, max_val], color='r')
    ax.plot(y_true, y_pred, 'g.')
    ax.grid()
    ax.set_xlabel('True distances', fontsize=12)
    ax.set_ylabel('Predicted distances', fontsize=12)
    ax.set_title('Scatter plot of true vs predicted distances', fontsize=15)
    # Save resulting figure
    fig.savefig('../data/models/{}/{}_scatter_{}_{}.png'.format(mod_name, mod_name, unseen_shuffled, length))
    print('\n\tScatter plot saved in ../data/models/{}/{}_scatter_{}_{}.png'.format(mod_name, mod_name,
                                                                                     unseen_shuffled, length))

def series_plot(y_true, y_pred, mod_name):
    #max_val = max(max(y_true), max(y_pred)) + 5
    length = len(y_true)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.plot(y_true.iloc[:, 0], y_true.iloc[:, 1], 'b-.')
    ax.plot(y_true.iloc[:, 0], y_pred, 'g-')
    ax.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Distances [m]', fontsize=12)
    ax.legend(['True', 'Predicted'], fontsize=12)
    ax.set_title('Series plot of true and predicted distances', fontsize=15)
    # Save resulting figure
    fig.savefig('../data/models/{}/{}_series_{}.png'.format(mod_name, mod_name, length))
    print('\n\tSeries plot saved in ../data/models/{}/{}_series_{}.png'.format(mod_name, mod_name, length))

def test_model_unseen(model, mod_name, n_frames, shuffled):
    video = '/V0420043'
    # Define parameters for the generator
    params = {'batch_size': 2**5,
              'shuffle': False,
              'X_reshape': model.layers[0].input_shape,
              'timestep': 10 if any(sb_str in mod_name for sb_str in ['rnn', 'lstm', 'gru']) else 0,
              'files' : {
                'file_X': '../data/clean/car_frames_norm.h5',
                'file_y': '../data/clean/car_distances_adjusted.h5',
                'group_X': video,
                'group_y': video
               }
             }

    # Total number of samples
    with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
        y = store_y[params['files']['group_y']]
        max_samples = y.shape[0]
    # Possibility to compute distaces for ALL frames
    n_frames = max_samples if n_frames == 0 else n_frames
    if shuffled:
        pred_idx = np.random.choice(max_samples, size=n_frames, replace=False)
    else:
        pred_idx = np.arange(n_frames)
    # Generators
    pred_generator = DataGenerator(pred_idx, **params)
    y_pred = model.predict_generator(pred_generator, verbose=1,
                                     use_multiprocessing=True, workers=2).ravel()
    y_true = y.iloc[pred_idx[-len(y_pred):], 1]
    print('\n\tResults on the evaluated frames:')
    print('\tMSE: {}'.format(keras.backend.eval(keras.losses.mean_squared_error(y_true, y_pred))))
    print('\tMAE: {}'.format(keras.backend.eval(keras.losses.mean_absolute_error(y_true, y_pred))))
    scatter_plot(y_true, y_pred, mod_name, unseen_shuffled='U' + ('S' if shuffled else ''))
    if not shuffled:
        # Plot both time series
        series_plot(y.iloc[pred_idx[-len(y_pred):], :], y_pred, mod_name)

def test_model_seen(model, mod_name, n_frames):
    # Define parameters for the generator
    params = {'batch_size': 2**5,
              'shuffle': False,
              'X_reshape': model.layers[0].input_shape,
              'timestep': 10 if any(sb_str in mod_name for sb_str in ['rnn', 'lstm', 'gru']) else 0,
              'files' : {
                'file_X': '../data/clean/car_frames_norm_shuffled.h5',
                'file_y': '../data/clean/car_distances_adjusted_shuffled.h5',
                'group_X': '/frames',
                'group_y': '/distances'
               }
             }

    # Total number of samples
    with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
        y = store_y[params['files']['group_y']]
        max_samples = y.shape[0]
    pred_idx = np.random.choice(max_samples, size=n_frames, replace=False)
    # Generators
    pred_generator = DataGenerator(pred_idx, **params)
    y_pred = model.predict_generator(pred_generator, verbose=1,
                                     use_multiprocessing=True, workers=2).ravel()
    y_true = y.iloc[pred_idx[-len(y_pred):], 1]
    print('\n\tResults on the evaluated frames:')
    print('\tMSE: {}'.format(keras.backend.eval(keras.losses.mean_squared_error(y_true, y_pred))))
    print('\tMAE: {}'.format(keras.backend.eval(keras.losses.mean_absolute_error(y_true, y_pred))))
    scatter_plot(y_true, y_pred, mod_name, unseen_shuffled='SS')

def prepare_test(models):
    while True:
        print('\n\t1. Load existing model.')
        print('\t2. Check model summary.')
        print('\t3. Plot model loss and metrics.')
        print('\t4. Test on unseen frames (V0420043).')
        print('\t5. Test on seen random frames.')
        print('\t6. Back to menu.')
        option = int(input('\n\tPlease enter your option: '))
        if option == 1:   model, mod_name = load_model(models)
        elif option == 2: model.summary()
        elif option == 3: loss_plot(model, mod_name)
        elif option == 4:
            n_frames = int(input('\n\tNumber of frames to test: '))
            shuffled = int(input('\n\tTest sequentially [1] or shuffled [2]?: ')) != 1
            test_model_unseen(model, mod_name, n_frames, shuffled)
        elif option == 5:
            n_frames = int(input('\n\tNumber of (random) frames to test: '))
            test_model_seen(model, mod_name, n_frames)
        elif option == 6: break
        else:
            print('\n\tInvalid option!')
            input('\n\tPress Enter to continue...')
    return


#################################################
#              HELPER FUNCTIONS                 #
#################################################

def load_model(models):
    print()
    while True:
        mod_name = input('\tModel file name [*.h5]: ')
        path = '../data/models/{}/{}.h5'.format(mod_name, mod_name)
        if os.path.exists(path): 
        #    mod_name = path.split('/')[-1][:-3]
            break
        else:
            print('\n\tWrong file name! -> {}'.format(path))
    do_compile = input('\n\tCompile model? [y/n]: ')
    model = models.load_model(path, True if do_compile == 'y' else False)
    return model, mod_name
    
def set_optimizer_options(models):
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print('\n\tOptions to change:')
        print('\n\t1. Optimizer: {}'.format(models.get_optimizer()))
        print('\t2. Loss: {}'.format(models.get_loss()))
        print('\t3. Metrics: {}'.format(models.get_metrics()))
        print('\t4. Learning rate: {}'.format(models.get_lr()))
        print('\t5. Back to menu.')
        option = int(input('\n\tPlease enter your option: '))
        value = input('\tValue: ')
        if   option == 1: models.set_optimizer(eval('keras.optimizers.' + value))
        elif option == 2: models.set_loss(value)
        elif option == 3: models.set_metrics(eval(value))
        elif option == 4: models.set_lr(float(value))
        elif option == 5: break
        else:
            print('\n\tInvalid option!')
            input('\n\tPress Enter to continue...')
            continue
    
def reload_models():
    reload(models_utils)
    models = modelUtils.Models()
    input('\n\tDone! Press Enter to continue...')
    return models

def memory_usage():
    print('\n\t- Total memory: {}GB'.format(psutil.virtual_memory().total >> 30))
    print('\t- Used memory: {}GB'.format(psutil.virtual_memory().used >> 30))
    print('\t- Free memory: {}GB'.format(psutil.virtual_memory().available >> 30))
    print('\t- Used memory: {}%'.format(psutil.virtual_memory().percent))
    input('\n\tPress Enter to continue...')
    
def disk_usage():
    print('\n\t- Total space: {}GB'.format(psutil.disk_usage('/').total >> 30))
    print('\t- Used space: {}GB'.format(psutil.disk_usage('/').used >> 30))
    print('\t- Free space: {}GB'.format(psutil.disk_usage('/').free >> 30))
    print('\t- Used space: {}%'.format(psutil.disk_usage('/').percent))
    input('\n\tPress Enter to continue...')


#################################################
#                    MENU                       #
#################################################

def run_playground():
    models = modelUtils.Models()
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print()
        print('\n\n\tDeep Playground ready!\n')
        print('\t1. Train models.')
        print('\t2. Test trained models.')
        print('\t3. Reload Models class.')
        print('\t4. Change optimizer options.')
        print('\t5. Check available memory.')
        print('\t6. Check disk usage.')
        print('\t7. Exit.')
        option = int(input('\n\tPlease enter your option: '))
        print('\tYou have entered: {}'.format(option))
        if   option == 1: prepare_training(models)
        elif option == 2: prepare_test(models)
        elif option == 3: models = reload_models()
        elif option == 4: set_optimizer_options(models)
        elif option == 5: memory_usage()
        elif option == 6: disk_usage()
        elif option == 7: 
            print('\n\n\tBye bye!\n')
            break
        else:
            print('\n\tInvalid option!')
            input('\n\tPress Enter to continue...')
            continue

if __name__ == "__main__":
    run_playground()
    