import numpy as np
import pandas as pd
import psutil, os, keras
from importlib import reload
from trainUtils import HistoryCallback, DataGenerator
import modelUtils


def train_models(mod_list, mod_names, epochs_list):
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
    train_idx = np.array(list(set(np.arange(n_samples)) - set(val_idx)))

    # Generators
    train_generator = DataGenerator(train_idx, **params)
    val_generator = DataGenerator(val_idx, **params)
    
    for idx, model in enumerate(mod_list):
        mod_name, n_epochs = mod_names[idx], epochs_list[idx]
        print('\n\n\t{}. Starting to train model {}...\n'.format(idx, mod_name))
        model.fit_generator(generator=train_generator,
                            validation_data=val_generator,
                            epochs=n_epochs, callbacks=[HistoryCallback(mod_name)],
                            use_multiprocessing=True, workers=2)
        model.save('../data/models/{}.h5'.format(mod_name))


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
            print(model.summary())
        print('\n\tModel added to queue.')
            
def load_model(models):
    print()
    while True:
        path = '../data/models/' + input('\tModel file name [*.h5]: ') + '.h5'
        if os.path.exists(path): 
            mod_name = path.split('/')[-1][:-3]
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


def run_playground():
    models = modelUtils.Models()
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print()
        print('\n\n\tDeep Playground ready!\n')
        print('\t1. Train models.')
        print('\t2. Load trained model.') # Run inference
        print('\t3. Reload Models class.')
        print('\t4. Change optimizer options.')
        print('\t5. Check available memory.')
        print('\t6. Check disk usage.')
        print('\t7. Exit.')
        option = int(input('\n\tPlease enter your option: '))
        print('\tYou have entered: {}'.format(option))
        if   option == 1: prepare_training(models)
        elif option == 2: model, mod_name = load_model(models)
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
    