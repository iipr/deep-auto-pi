import numpy as np
import pandas as pd
import psutil, os, keras, time
from importlib import reload
from trainUtils import HistoryCallback, DataGenerator
import modelUtils
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


#################################################
#             Global parameters                 #
#################################################

MODEL_PATH = '../data/models/'
FRAMES = '../data/clean/car_frames_norm.h5'
DIST = '../data/clean/car_distances_adjusted.h5'
FRAMES_GROUP = ''
DIST_GROUP = ''
FRAMES_SHUFFLED = '../data/clean/car_frames_norm_shuffled.h5'
DIST_SHUFFLED = '../data/clean/car_distances_adjusted_shuffled.h5'
FRAMES_SHUFFLED_GROUP = '/frames'
DIST_SHUFFLED_GROUP = '/distances'
BATCH_SIZE = 2**5
BATCH_SIZE_TS = 50
SHUFFLE_DATA_GEN = False
TIMESTEP = 3 # <- for LSTMs
TRAIN_SPLIT = 0.9
VAL_SPLIT = 1 - TRAIN_SPLIT
UNSEEN_VID = '/V0420043'
RNN_NAMES = ['rnn', 'lstm', 'gru', 'time_distributed']
CAT_MAP = {
     'cat_00': 0,
     'cat_01': 1,
     'cat_02': 2,
     'cat_03': 3,
     'cat_04': 4,
     'cat_05': 5,
     'cat_06': 6,
     'cat_07': 7,
     'cat_08': 8,
     'cat_09': 9,
     'cat_10': 10,
     'cat_11': 11
}

#################################################
#                  TRAINING                     #
#################################################

def train_sequentially(model, n_epochs):
    '''For the case of LSTMs.'''
    # Define parameters for the generator
    params = {'batch_size': BATCH_SIZE_TS if has_rnn_layer(model) else BATCH_SIZE,
              'shuffle': SHUFFLE_DATA_GEN,
              'X_reshape': model.layers[0].input_shape,
              'timestep': TIMESTEP if has_rnn_layer(model) else 0,
              'files' : {
                'file_X': FRAMES,
                'file_y': DIST.replace('.h5', '_cat.h5') if 'cat' in model.name else DIST,
                'group_X': FRAMES_GROUP,
                'group_y': DIST_GROUP
               }
             }
    # Extract video names
    with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
        videos = [video for video in store_y]
    for epoch in range(n_epochs):
        for idx, video in enumerate(videos):
            # Leave the Tesla-Mercedes video as a test set
            if video in [UNSEEN_VID]:
                continue
            # Update params with the new video name
            params['files']['group_X'], params['files']['group_y'] = video, video
            # Total number of samples
            with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
                n_samples = store_y[video].shape[0]
            train_split = TRAIN_SPLIT
            # Indexes for train and validation
            train_idx = np.arange(n_samples)[0:int(n_samples * train_split)]
            val_idx = np.arange(n_samples)[int(n_samples * train_split):n_samples]
            # Generators
            train_generator = DataGenerator(train_idx, **params)
            val_generator = DataGenerator(val_idx, **params)

            model.fit_generator(generator=train_generator,
                                validation_data=val_generator, shuffle=False,
                                initial_epoch=epoch, epochs=epoch+1,
                                callbacks=[HistoryCallback(mod_name=model.name, video=video,
                                                           log_file=get_path(MODEL_PATH, model.name, '_logs.txt'))],
                                use_multiprocessing=True, workers=2)
            model.reset_states()
    return model, params

def train_shuffled(model, n_epochs):
    '''For the case of other NN (like CNNs).'''
    # Define parameters for the generator
    params = {'batch_size': BATCH_SIZE_TS if has_rnn_layer(model) else BATCH_SIZE,
              'shuffle': SHUFFLE_DATA_GEN,
              'X_reshape': model.layers[0].input_shape,
              'timestep': TIMESTEP if has_rnn_layer(model) else 0,
              'files' : {
                'file_X': FRAMES_SHUFFLED,
                'file_y': DIST_SHUFFLED.replace('.h5', '_cat.h5') if 'cat' in model.name else DIST_SHUFFLED,
                'group_X': FRAMES_SHUFFLED_GROUP,
                'group_y': DIST_SHUFFLED_GROUP
               }
             }
    # Total number of samples
    with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
        n_samples = store_y[params['files']['group_y']].shape[0]
    validation_split = VAL_SPLIT
    # Indexes of the validation set
    val_idx = np.random.choice(np.arange(n_samples), replace=False,
                               size=int(np.floor(n_samples * validation_split)))
    # Indexes of the training set
    train_idx = np.array(list(set(np.arange(n_samples)) - set(val_idx)))
    # Generators
    train_generator = DataGenerator(train_idx, **params)
    val_generator = DataGenerator(val_idx, **params)

    model.fit_generator(generator=train_generator,
                        validation_data=val_generator, 
                        epochs=n_epochs, #shuffle=False,
                        callbacks=[HistoryCallback(mod_name=model.name, 
                                                   log_file=get_path(MODEL_PATH, model.name, '_logs.txt'))],
                        use_multiprocessing=True, workers=2)
    return model, params

def train_models(mod_list, epochs_list, models):
    for idx, model in enumerate(mod_list):
        n_epochs = epochs_list[idx]
        if not os.path.exists('{}{}/'.format(MODEL_PATH, model.name)):
            os.mkdir('{}{}/'.format(MODEL_PATH, model.name))
        print('\n\n\t{}. Starting to train model {}...\n'.format(idx, model.name))
        if has_rnn_layer(model):
            model, params = train_sequentially(model,n_epochs)
        else:
            model, params = train_shuffled(model, n_epochs)
        model.save(get_path(MODEL_PATH, model.name, '.h5'))
        # For reproducibility:
        save_meta(model, n_epochs, params, models)

def prepare_training(models):
    mod_list, epochs_list = [], []
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
        elif option == 2: model = load_model(models)
        elif option == 3: print('\n\t{}'.format([(mod_list[i].name, epochs_list[i]) 
                                                 for i in range(len(mod_list))])); continue
        elif option == 4: train_models(mod_list, epochs_list, models); break
        elif option == 5: break
        else:
            print('\n\tInvalid option!')
            input('\n\tPress Enter to continue...')
            continue
        model.name = input('\n\tPlease enter a new model name: ')
        mod_list.append(model)
        epochs_list.append(int(input('\n\tNumber of epochs: ')))
        if input('\n\tSee model summary? [y/n]: ') == 'y':
            model.summary()
        print('\n\tModel added to queue.')


#################################################
#                  TESTING                      #
#################################################

def loss_plot(mod_name):
    # Load history file
    hist = pd.read_csv(get_path(MODEL_PATH, mod_name, '_hist.csv'))
    days = hist['duration [s]'].sum() // (24 * 3600)
    print('\n\tTotal training time for model {} was {} days and {}'.format(mod_name, days,
        time.strftime('%H:%M:%S', time.gmtime(hist['duration [s]'].sum()))))
    # Plot loss and metrics
    fig, axs = plt.subplots(2,1)
    fig.set_size_inches(15, 10)
    # Define names of losses and metrics
    loss = 'categorical_crossentropy' if 'cat' in mod_name else 'mean_squared_error'
    metric = 'acc' if 'cat' in mod_name else 'mean_absolute_error'
    loss_names = {
        'mean_squared_error': 'Mean Squared Error',
        'mean_absolute_error': 'Mean Absolute Error',
        'acc': 'Accuracy',
        'categorical_crossentropy': 'Categorical Crossentropy'
    }
    axs[0].plot(hist[metric], linewidth=1.75)
    axs[0].plot(hist['val_' + metric], linewidth=1.75)
    axs[0].xaxis.set_tick_params(labelsize=20)
    axs[0].yaxis.set_tick_params(labelsize=20)
    axs[0].legend(['Training set', 'Validation set'], fontsize=25)
    axs[0].set_title(loss_names[metric], fontsize=30)
    #axs[0].set_yscale('log')
    axs[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[0].grid()
    axs[1].plot(hist['loss'], linewidth=1.75)
    axs[1].plot(hist['val_loss'], linewidth=1.75)
    axs[1].xaxis.set_tick_params(labelsize=20)
    axs[1].yaxis.set_tick_params(labelsize=20)
    axs[1].legend(['Training set', 'Validation set'], fontsize=25)
    axs[1].set_title(loss_names[loss], fontsize=30)
    #axs[1].set_yscale('log')
    axs[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs[1].grid()
    fig.tight_layout()
    # Save resulting figure
    plt_path = get_path(MODEL_PATH, mod_name, '_loss.pdf')
    fig.savefig(plt_path, bbox_inches='tight')
    print('\n\tLoss plots saved in {}'.format(plt_path))

def scatter_plot(y_true, y_pred, mod_name, video=''):
    max_val = max(max(y_true), max(y_pred)) + 5
    length = len(y_true)
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(y_true.values.reshape(-1, 1), y_pred)
    # Make predictions using the testing set
    line = regr.predict(y_true.values.reshape(-1, 1))
    # Make the plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    ax.plot(y_true, line, color='royalblue')
    ax.plot([0, max_val], [0, max_val], color='r')
    ax.plot(y_true, y_pred, 'g.')
    ax.grid()
    ax.legend(['LR of True vs Predicted', 'y(x) = x', 'True vs Predicted'],
              fontsize=25)
    ax.set_xlabel('True distances', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_ylabel('Predicted distances', fontsize=30)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_title('Scatter plot of true vs predicted distances\nR^2 = {:0.3}, MSE = {:0.5}'.format(
        r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred)), fontsize=30)
    # Save resulting figure
    plt_path = get_path(MODEL_PATH, mod_name, '_scatter_{}_{}.pdf'.format(video, length))
    fig.savefig(plt_path, bbox_inches='tight')
    print('\n\tScatter plot saved in {}'.format(plt_path))

def series_plot(y_true, y_pred, mod_name):
    length = len(y_true)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    ax.plot(y_true.iloc[:, 0], y_true.iloc[:, 1], 'b-.')
    ax.plot(y_true.iloc[:, 0], y_pred, 'g-')
    ax.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel('Timestamp', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_ylabel('Distances [m]', fontsize=30)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.legend(['True', 'Predicted'], fontsize=25)
    ax.set_title('Series plot of true and predicted distances', fontsize=30)
    # Save resulting figure
    plt_path = get_path(MODEL_PATH, mod_name, '_series_{}.pdf'.format(length))
    fig.savefig(plt_path, bbox_inches='tight')
    print('\n\tSeries plot saved in {}'.format(plt_path))

def test_single_video(model, n_frames, shuffled, video=UNSEEN_VID):
    # Define parameters for the generator
    params = {'batch_size': BATCH_SIZE_TS if has_rnn_layer(model) else BATCH_SIZE,
              'shuffle': SHUFFLE_DATA_GEN,
              'X_reshape': model.layers[0].input_shape,
              'timestep': TIMESTEP if has_rnn_layer(model) else 0,
              'files' : {
                'file_X': FRAMES,
                'file_y': DIST.replace('.h5', '_cat.h5') if 'cat' in model.name else DIST,
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
        y_true = y.iloc[pred_idx, 1:]
    elif params['timestep'] > 0:
        pred_idx = np.arange(n_frames)
        start = params['timestep'] - 1
        end = start + n_frames - (n_frames % params['batch_size'])
        y_true = y.iloc[start:end, 1:]
    else:
        pred_idx = np.arange(n_frames)
        y_true = y.iloc[pred_idx, 1:]
    # Generator
    pred_generator = DataGenerator(pred_idx, **params)
    if 'cat' in model.name:
        test_for_categories(model, pred_generator, y_true)
    else:
        y_true = y_true.iloc[:, 0]
        y_pred = model.predict_generator(pred_generator, verbose=1,
                                         use_multiprocessing=True, workers=2).ravel()
        print('\n\tResults on the evaluated frames:')
        print('\tMSE: {}'.format(keras.backend.eval(keras.losses.mean_squared_error(y_true, y_pred))))
        print('\tMAE: {}'.format(keras.backend.eval(keras.losses.mean_absolute_error(y_true, y_pred))))
        scatter_plot(y_true, y_pred, model.name, video=video[1:])
        if not shuffled: # Plot both time series
            series_plot(y.iloc[pred_idx[-len(y_pred):], :], y_pred, model.name)

def test_shuffled_frames(model, n_frames):
    # Define parameters for the generator
    params = {'batch_size': BATCH_SIZE_TS if has_rnn_layer(model) else BATCH_SIZE,
              'shuffle': SHUFFLE_DATA_GEN,
              'X_reshape': model.layers[0].input_shape,
              'timestep': TIMESTEP if has_rnn_layer(model) else 0,
              'files' : {
                'file_X': FRAMES_SHUFFLED,
                'file_y': DIST_SHUFFLED.replace('.h5', '_cat.h5') if 'cat' in model.name else DIST_SHUFFLED,
                'group_X': FRAMES_SHUFFLED_GROUP,
                'group_y': DIST_SHUFFLED_GROUP
               }
             }

    # Total number of samples
    with pd.HDFStore(params['files']['file_y'], mode='r') as store_y:
        y = store_y[params['files']['group_y']]
    max_samples = y.shape[0]
    pred_idx = np.random.choice(max_samples, size=n_frames, replace=False)
    y_true = y.iloc[pred_idx, 1:]
    # Generator
    pred_generator = DataGenerator(pred_idx, **params)
    if 'cat' in model.name:
        test_for_categories(model, pred_generator, y_true)
    else:
        y_pred = model.predict_generator(pred_generator, verbose=1,
                                         use_multiprocessing=True, workers=2).ravel()
        print('\n\tResults on the evaluated frames:')
        print('\tMSE: {}'.format(keras.backend.eval(keras.losses.mean_squared_error(y_true, y_pred))))
        print('\tMAE: {}'.format(keras.backend.eval(keras.losses.mean_absolute_error(y_true, y_pred))))
        scatter_plot(y_true, y_pred, model.name, video='seen')

def test_for_categories(model, pred_generator, y_true):
    y_pred = model.predict_generator(pred_generator, verbose=1,
                                      use_multiprocessing=True, workers=2)
    #print('\n\tWith the test set, losses are {:0.5}, and accuracy is {:0.5}'.format(loss, acc))
    print('\n\tTop 1 accuracy: {}'.format(
        keras.backend.eval(keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1))))
    print('\tTop 3 accuracy: {}'.format(
        keras.backend.eval(keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))))
    print('\tConfusion matrix:')
    print(confusion_matrix(y_true=y_true.idxmax(axis=1).map(CAT_MAP), y_pred=y_pred.argmax(axis=1)))
    ROC_plot(model.name, y_true, y_pred)

def ROC_plot(mod_name, y_true, y_pred):
    # ROC curves
    n_classes = y_true.shape[1]
    half = n_classes // 2
    fpr, tpr, roc_auc = {}, {}, {}
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true.iloc[:, c], y_pred[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(24, 12)
    for c in range(n_classes):
        ax = axs[0] if c < 5 else axs[1]
        ax.plot(fpr[c], tpr[c])
    axs[0].set_title('ROC curve for the first {} classes'.format(half), fontsize=25)
    axs[0].set_xlabel('False Positive Rate', fontsize=25)
    axs[0].xaxis.set_tick_params(labelsize=15)
    axs[0].set_ylabel('True Positive Rate', fontsize=25)
    axs[0].yaxis.set_tick_params(labelsize=15)
    axs[0].legend(['Class {}, AUC = {:.03}'.format(c, roc_auc[c]) for c in range(half)],
                  fontsize=20)
    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].grid()
    axs[1].set_title('ROC curve for the last {} classes'.format(half), fontsize=25)
    axs[1].set_xlabel('False Positive Rate', fontsize=25)
    axs[0].xaxis.set_tick_params(labelsize=15)
    axs[1].set_ylabel('True Positive Rate', fontsize=25)
    axs[0].yaxis.set_tick_params(labelsize=15)
    axs[1].legend(['Class {}, AUC = {:.03}'.format(c, roc_auc[c]) for c in range(half, n_classes)],
                  fontsize=20)
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].grid()
    # Save resulting figure
    plt_path = get_path(MODEL_PATH, mod_name, '_roc.pdf')
    fig.savefig(plt_path, bbox_inches='tight')
    print('\n\tROC plot saved in {}'.format(plt_path))


def prepare_test(models):
    while True:
        print('\n\t1. Load existing model.')
        print('\t2. Check model summary.')
        print('\t3. Plot model loss and metrics.')
        print('\t4. Test on single video.')
        print('\t5. Test on (random) seen frames.')
        print('\t6. Back to menu.')
        option = int(input('\n\tPlease enter your option: '))
        if option == 1:   model = load_model(models)
        elif option == 2: model.summary()
        elif option == 3: loss_plot(model.name)
        elif option == 4:
            video = input('\n\tTest on unseen video [{}] or introduce video name: '.format(UNSEEN_VID))
            video = UNSEEN_VID if video == '' else '/' + video
            n_frames = int(input('\n\tNumber of frames to test: '))
            shuffled = int(input('\n\tTest sequentially [1] or shuffled [2]?: ')) != 1
            test_single_video(model, n_frames, shuffled, video=video)
        elif option == 5:
            n_frames = int(input('\n\tNumber of (random) frames to test: '))
            test_shuffled_frames(model, n_frames)
        elif option == 6: break
        else:
            print('\n\tInvalid option!')
            input('\n\tPress Enter to continue...')
    return


#################################################
#              HELPER FUNCTIONS                 #
#################################################

def has_rnn_layer(model):
    for layer in model.layers:
        if any(sb_str in layer.name for sb_str in RNN_NAMES):
            return True
    return False

def get_path(path, name, ext):
    return path + name + '/' + name + ext

def load_model(models):
    mod_list = sorted([m for m in os.listdir(MODEL_PATH) if m[0] != '.' and m not in ['old', 'models.txt']])
    print('\n\tAvailable models are {}'.format(mod_list))
    while True:
        mod_name = input('\tModel file name [*.h5]: ')
        mod_path = get_path(MODEL_PATH, mod_name, '.h5')
        if os.path.exists(mod_path):
            break
        else:
            print('\n\tWrong file name! -> {}'.format(mod_path))
    do_compile = input('\n\tCompile model? [y/n]: ')
    model = models.load_model(mod_path, True if do_compile == 'y' else False)
    return model

def save_meta(model, n_epochs, params, models):
    width = 80
    with open(MODEL_PATH + 'models.txt', 'a+') as models_txt:
        models_txt.write('_' * width)
        models_txt.write('\n\nMODEL {}\n'.format(model.name))
        models_txt.write('\nn_epochs: {}'.format(n_epochs))
        models_txt.write('\nTRAIN_SPLIT: {}'.format(TRAIN_SPLIT))
        models_txt.write('\nbatch_size: {}'.format(params['batch_size']))
        models_txt.write('\nshuffle: {}'.format(params['shuffle']))
        models_txt.write('\nX_reshape: {}'.format(params['X_reshape']))
        models_txt.write('\ntimestep: {}'.format(params['timestep']))
        models_txt.write('\nfiles: \n    {}\n    {}\n    {}\n    {}'.format(
            *[item for item in params['files'].items()]))
        models_txt.write('\noptimizer: {}'.format(models.get_optimizer()))
        models_txt.write('\nloss: {}'.format(models.get_loss()))
        models_txt.write('\nmetrics: {}'.format(models.get_metrics()))
        models_txt.write('\nlr: {}\n\n'.format(models.get_lr()))
        model.summary(line_length=width, print_fn=lambda x: models_txt.write(x + '\n'))
        models_txt.write('\n\n')

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
    reload(modelUtils)
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
    