import os
import sys
# import datetime
import numpy as np
# import pandas as pd
import joblib
import pandas as pd
import csv

import tensorflow as tf
from tensorflow.keras import models, callbacks
# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras import applications
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Change this to the location of the database directories
DB_DIR = '/store/datasets/covid/audiosl/'

# Import databases
sys.path.insert(1, DB_DIR)

from MIT_models import choose_multimodel, choose_model, train_model_full_steps
from db_utils import poissonw_noise, import_language_PN, get_sentiment_data, get_librispeech_wakeword, \
    normalize_dataset_PN, job_maker, chunk_maker_random, combine_3_chunks, delete_files_from_job, balance_dataset, \
    import_language_PN_white_noise, add_silence_to_dataset, import_language_PN_STFT


def normalize_dataset(X):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    print('Normalizing: mean=', mean, 'std=', std)

    return X



def reshape_dataset(X):
    """Reshape dataset for Convolution."""
    # num_pixels = X.shape[1]*X.shape[2]

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1).astype('float32')

    return X


def split_dataset(X, Y, test_size, val_split):
    """Returns training, validation and testing dataset in random order. Sets of data may not be balanced"""
    import random
    rs = random.randint(1, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=test_size, random_state=rs)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False, test_size=val_split)

    return X_train, y_train, X_test, y_test, X_val, y_val


def arrange_dataset(X_pos, X_neg, files_p, files_n, state, test_size=0.2, val_split=0.2, pre_shuffle=True):
    """Returns training, validation, and testing dataset with labeling in a [0,1,0,1,0,1,...] fashion. Every set of data will be balanced"""

    state = int(state)

    assert len(X_pos) == len(X_neg)
    l = len(X_pos)
    # X = np.insert(X_pos, np.arange(len(X_neg)), X_neg, axis=0)
    # Y = np.insert(Y_pos, np.arange(len(Y_neg)), Y_neg)

    if pre_shuffle:
        X_pos, files_p = shuffle(X_pos, files_p, random_state=1)
        X_neg, files_n = shuffle(X_neg, files_n, random_state=2)

    # Apply
    while state > l:
        state = int(state / 2)
        print("state bigger than dataset, reduced in half: state = ", state)
    if state % 2 != 0:
        state -= 1
        print("making state even by substracting 1: state =", state)

    # We will set the state element as the index 0 element
    X_pos = np.concatenate((X_pos[state:, :, :, :], X_pos[:state, :, :, :]), axis=0)
    X_neg = np.concatenate((X_neg[state:, :, :, :], X_neg[:state, :, :, :]), axis=0)

    files_p = files_p[state:] + files_p[:state]
    files_n = files_n[state:] + files_n[:state]

    # TRAIN AND TEST SPLIT
    target_test = int(l * test_size)
    target_val = int(l * val_split)
    # positives:
    X_test_p = X_pos[0:target_test, :, :, :]
    files_test_p = files_p[0:target_test]
    X_val_p = X_pos[target_test:target_test + target_val, :, :, :]
    files_val_p = files_p[target_test:target_test + target_val]
    X_train_p = X_pos[target_test + target_val:, :, :, :]
    files_train_p = files_p[target_test + target_val:]

    # negatives
    X_test_n = X_neg[0:target_test, :, :, :]
    files_test_n = files_n[0:target_test]
    X_val_n = X_neg[target_test:target_test + target_val, :, :, :]
    files_val_n = files_n[target_test:target_test + target_val]
    X_train_n = X_neg[target_test + target_val:, :, :, :]
    files_train_n = files_n[target_test + target_val:]

    # Mix in order: [0, 1, 0, 1, ...]
    X_train = np.insert(X_train_p, np.arange(len(X_train_n)), X_train_n, axis=0)
    Y_train = np.insert(np.ones(len(X_train_p)), np.arange(len(X_train_n)), np.zeros(len(X_train_n)))

    X_val = np.insert(X_val_p, np.arange(len(X_val_n)), X_val_n, axis=0)
    Y_val = np.insert(np.ones(len(X_val_p)), np.arange(len(X_val_n)), np.zeros(len(X_val_n)))

    X_test = np.insert(X_test_p, np.arange(len(X_test_n)), X_test_n, axis=0)
    Y_test = np.insert(np.ones(len(X_test_p)), np.arange(len(X_test_n)), np.zeros(len(X_test_n)))

    # Create dictionary with all the files used in each set
    files = {}
    files['train positives'] = files_train_p
    files['train negatives'] = files_train_n
    files['test positives'] = files_test_p
    files['test negatives'] = files_test_n
    files['validation positives'] = files_val_p
    files['validation negatives'] = files_val_n

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, files


def train_basemodel(dataset, SAVE_DIR, monitor, model_type="ResNet50", save_name='t5_c100', epochs=100, lr=0.001,
                    language1='english', language2='russian', test_size=0.2, val_split=0.2, batch_size=10, patience=20):
    """Trains biomarker model to be later concatenated at the multimodel"""

    # Create directory to save results
    if dataset not in os.listdir(SAVE_DIR):
        os.mkdir(SAVE_DIR + dataset)

    SAVE_DIR += dataset + '/'

    # Load datasets
    if dataset == 'Librispeech':
        db_path = '/store/datasets/aux'
        # X, Y = joblib.load('/store/datasets/jobs/XY_Libri_t5_repeated_c100.job')
        X, Y = get_librispeech_wakeword(db_path, time=2, wakeword='THEM', cepstrum_dimension=200)

        # Repeat the 1 second 5 times
        # X = []
        # for x in X_1:
        #    X.append(np.concatenate((x, x, x, x, x), axis=0))

        joblib.dump((X, Y), '/store/datasets/jobs/XY_Libri_t2_c200.job')

    elif dataset == 'Sentiment':
        X, Y = joblib.load(DB_DIR + '../../jobs/XY_sentiment_5s_100.job')
        # X, Y = get_sentiment_data()
        # joblib.dump((X,Y), '/store/datasets/jobs/XY_sentiment_5s_100.job')

    elif dataset == 'Languages':
        src = '/store/datasets/covid/audiosl/bioM_languages/'
        dst = '/store/datasets/jobs/'
        _ = job_maker('spanish', 'spanish_cep100', src, dst)
        _ = job_maker('english-AW', 'english-AW_cep100', src, dst)

        # X, Y, pos_files, neg_files = import_language_bioM('%s_cep100' % language1, '%s_cep100' % language2, 'english-AW',
        #                                                  'spanish')
        # files_used = pos_files.append(neg_files)
        # print('files used length', len(files_used), np.shape(files_used))

    # X = normalize_dataset(X)
    X = reshape_dataset(X)

    # import datetime
    # time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=test_size, random_state=13)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False, test_size=val_split)

    print("training: ", dataset)

    # Apply poisson
    X_masked = []
    for i in X_train:
        X_masked.append(poissonw_noise(i))

    X_train = np.concatenate((X_train, X_masked))
    y_train = np.concatenate((y_train, y_train))

    print('data ready', ' X_train = ', np.shape(X_train), 'Y_train = ', np.shape(y_train))
    # Create and compile model
    opt = Adam(lr=lr)
    if dataset == 'Sentiment':
        model = choose_model(X_train.shape[1:], classes=8, model_name=model_type)  # By default: model_name = Resnet50
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    else:
        model = choose_model(X_train.shape[1:], classes=1, model_name=model_type)  # By default: model_name = Resnet50
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    """Set a checkpoint to save the model whenever the validation accuracy improves"""
    model_name = '%s_%s.h5' % (dataset, save_name)
    checkpoint = callbacks.ModelCheckpoint(SAVE_DIR + model_name, monitor=monitor, verbose=1,
                                           save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch")
    early = callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=1, mode='auto')

    cb = [checkpoint, early]

    """Fit model"""
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=batch_size, verbose=1, callbacks=cb)

    best_model = models.load_model(SAVE_DIR + model_name)
    scores = best_model.evaluate(X_test, y_test, verbose=2)

    model_path = SAVE_DIR + '%i_%s' % (int(scores[1] * 100), model_name)
    os.rename(SAVE_DIR + model_name, model_path)

    return model_path


def main(pre_train=False, full_paths=False, multi_input=False, add_librispeech=False):
    """Training and testing a full model.
    - Set pre_train = True to train base models
    - Set multi_input = True to create a 3 Resnet model, otherwise just one Resnet.
    - Set full_paths = True to save the full paths to
    """
    # import variables from parameters.json file
    import json
    with open('parameters.json') as f:
        variables = json.load(f)

    # Establish visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = variables['GPUs']

    # Make experiment dir
    SAVE_DIR = "/store/experiments/covid/"
    EXP = variables['EXP']  # Folder to be created under experiment name in SAVE_DIR
    save_name = variables['save_name']  # Subdirectory to be created in EXP

    if EXP not in os.listdir(SAVE_DIR):
        os.mkdir(SAVE_DIR + EXP)
    if save_name not in os.listdir(SAVE_DIR + EXP):
        os.mkdir(SAVE_DIR + EXP + '/' + save_name)

    # Results will be saved here:
    save_path = SAVE_DIR + EXP + '/' + save_name + '/'

    # Parameters used:
    epochs = variables['epochs']  # Number of repetitions of the training
    patience = variables['patience']  # Epochs waited without improvements in the monitored value before early stopping the training
    monitor = variables['monitor']  # Monitored value
    lr = variables['lr']  # Learning rate
    opt = variables['opt']
    steps = variables['steps']  # Batch size

    # Apply exp decay if desired
    # exp_decay = variables['exp_decay']
    # if exp_decay:
    #    lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=10000, decay_rate=0.95, staircase=True)

    # Train base models
    if pre_train:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            path_libri = train_basemodel('Librispeech', save_path, model_type="ResNet50",
                                         monitor='val_loss', epochs=epochs, lr=lr, save_name= "RN50_t5_c100", batch_size=50, patience=10)

            path_sentiment = train_basemodel('Sentiment', SAVE_DIR + EXP + '/', model_type="ResNet50",
                                             monitor='val_loss', epochs=epochs, lr=0.001, save_name="RN50-new_t5_c100", batch_size=10, patience=50)

            # language2 can be a list of languages
            # Possible languages : 'english-AW' 'russian' 'catalan' 'spanish'
            path_languages = train_basemodel('Languages', '/store/experiments/covid/new_basemodels/', monitor='val_loss',
                                             epochs=epochs, lr=lr, language1='english-AW', language2='spanish', batch_size=100, patience=10)

    # Or use pretrained models
    else:
        # path_languages = '/store/models/bio_markers/72CategoricalLanguages_t5_cep100.h5'
        path_libri = variables['path_libri']
        path_sentiment = variables['path_sentiment']
        path_languages = variables['path_languages']

    # Import dataset
    # 'asymptomatic-official-web-cat-eng-spa' 'main'
    load_type = variables['load_type']
    language = variables['language']

    if load_type == 'import':
        # Import mfcc
        print('importing dataset:', language)
        X_pos, X_neg, files_used_p, files_used_n = import_language_PN(language, 100)
        print('files positives:', len(files_used_p))
        print('files negatives:', len(files_used_n))

        if full_paths:
            files_used_p = ['/store/datasets/covid/audiosl/languages/%s/positives/' % language + f for f in
                            files_used_p]
            files_used_n = ['/store/datasets/covid/audiosl/languages/%s/negatives/' % language + f for f in
                            files_used_n]

        normalization_type = "z"

    elif load_type == 'job':
        print('loading job files')
        job1 = variables['job_p']
        job2 = variables['job_n']

        X_pos = np.asarray(joblib.load(job1))
        X_neg = np.asarray(joblib.load(job2))

        print('shape of positive files: ', X_pos.shape)
        print('shape of negative files: ', X_neg.shape)

        path_p = os.path.dirname(job1) + '/'
        path_n = os.path.dirname(job2) + '/'

        # Load list with names of the files
        csv1 = [f for f in os.listdir(path_p) if f.find('positives.csv') != -1]
        with open(path_p + csv1[0], 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            files_used_p = list(reader)[0]

        csv2 = [f for f in os.listdir(path_n) if f.find('negatives.csv') != -1]
        with open(path_n + csv2[0], 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            files_used_n = list(reader)[0]

        if full_paths:
            # Positives
            if path_p.find('entiment')!=-1:
                files_used_p = ['/store/datasets/aux/all_sentiment_links/' + f for f in
                                files_used_p]
            elif path_p.find('ibrispeech')!=-1:
                files_used_p = ['/store/datasets/aux/all_libriseech_links/' + f for f in
                                files_used_p]
            else:
                files_used_p = ['/store/datasets/covid/audiosl/languages/%s/positives/' % language + f for f in
                                files_used_p]
            # Negatives
            if path_n.find('entiment')!=-1:
                files_used_n = ['/store/datasets/aux/all_sentiment_links/' + f for f in
                                files_used_n]
            elif path_n.find('ibrispeech')!=-1:
                files_used_n = ['/store/datasets/aux/all_libriseech_links/' + f for f in
                                files_used_n]
            else:
                files_used_n = ['/store/datasets/covid/audiosl/languages/%s/negatives/' % language + f for f in
                                files_used_n]

            normalization_type = "max"

        normalization_type = "z"

    else:
        print('WRONG LOAD TYPE!')
        X_pos, X_neg, files_used_p, files_used_n = (None, None, None, None)
        exit()


    # If there are some files that we want to delete
    # X_pos, files_used_p = delete_files_from_job(X_pos, files_used_p, '/store/datasets/covid/audiosl/languages/%s/defective_files_p.csv' % language)
    # X_neg, files_used_n = delete_files_from_job(X_neg, files_used_n,  '/store/datasets/covid/audiosl/languages/%s/defective_files_n.csv' % language)

    X_pos, X_neg, files_used_p, files_used_n = balance_dataset(X_pos, X_neg, files_used_p, files_used_n)

    #print(files_used_n)
    if add_librispeech:
        print('ADDING LIBRISPEECH')
        X_neg, files_used_n = add_librispeech_to_dataset(X_neg, files_used_n, load_type, full_paths, 0.5)

    #print(files_used_n)

    # Normalize dataset - Same normalization for positives and negatives
    X_pos, X_neg, normalization = normalize_dataset_PN(X_pos, X_neg, normalization_type)

    with open(save_path + 'normalization_parameters.json', "w") as f:
        json.dump(normalization, f)

    # reshape dataset
    X_pos = reshape_dataset(X_pos)
    X_neg = reshape_dataset(X_neg)

    # Split data
    test_size = variables['test_size']
    val_split = variables['val_split']
    state = variables['state']
    X_train, y_train, X_val, y_val, X_test, y_test, files = arrange_dataset(X_pos, X_neg, files_used_p, files_used_n,
                                                                            state, test_size, val_split, pre_shuffle=True)

    # Alternatively:
    # X = np.concatenate((X_pos, X_neg), axis=0)
    # Y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))
    # X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, Y, test_size, val_split)

    # Save files used to json
    with open(save_path + 'files_used.json', "w") as f:
        json.dump(files, f)

    print('saved files_used.json to:', save_path + 'files_used.json')

    # Apply poisson to training dataset
    X_masked_p = []
    for i in X_train:
        X_masked_p.append(poissonw_noise(i))

    # Apply white noise to the training set:
    #X_masked, y_masked = import_language_PN_white_noise(language,
    #                                                   files['train positives'][:int(len(files['train positives'])*0.05)],
    #                                                  files['train negatives'][:int(len(files['train negatives'])*0.05)])

    # X_masked, y_masked = import_language_PN_white_noise(language, files['train positives'], files['train negatives'])
    # X_masked = (X_masked - mean) / std
    # X_masked = reshape_dataset(X_masked)

    X_train = np.concatenate((X_train, X_masked_p), axis=0)
    y_train = np.concatenate((y_train, y_train))

    # Check that data has the desired shape
    print('training', np.shape(X_train), np.shape(y_train))
    # print(y_train)
    print('testing', np.shape(X_test), np.shape(y_test))
    # print(y_test)
    print('validation', np.shape(X_val), np.shape(y_val))
    # print(y_val)

    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=1)
    X_val, y_val = shuffle(X_val, y_val, random_state=2)
    X_test, y_test = shuffle(X_test, y_test, random_state=3)

    print('data ready:', 'X_train = ', np.shape(X_train), 'Y_train = ', np.shape(y_train))

    # Concatenate dataset to be read by the multi-model
    if multi_input:
        X_train = [X_train, X_train, X_train]
        X_val = [X_val, X_val, X_val]
        X_test = [X_test, X_test, X_test]

    # Synchronise GPUs and distribute memory
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # Create model with loaded weights from paths:
        if multi_input:
            model = choose_multimodel(np.array(X_train[0][0]).shape, path_languages, path_libri, path_sentiment)
        else:
            model = choose_model(X_train[0].shape)
        #print(model.summary())

        _, model_path = train_model_full_steps(model, X_train, y_train, save_path, language,
                                               (X_val, y_val), monitor, epochs, steps, opt, lr, patience)

        # Test model
        trained_model = models.load_model(model_path)
        scores = trained_model.evaluate(X_test, y_test, verbose=2)

    # Rename model:
    model_path2 = save_path + '%s_%i.h5' % (language, int(scores[1] * 100))
    os.rename(model_path, model_path2)
    os.rename(save_path + "model_history.csv", save_path + '%s_%i_history.csv' % (language, int(scores[1] * 100)))

    print("model saved to: ", model_path2)

    # Save results to json
    sensitivity = format(1 - scores[2] / len(y_test[y_test == 1]), '.2f')
    specificity = format(1 - scores[3] / len(y_test[y_test == 0]), '.2f')
    results = {}
    results['model_name'] = '%i_%s.h5' % (int(scores[1] * 100), language)
    results['loss'] = scores[0]
    results['accuracy'] = scores[1]
    results['sensitivity'] = sensitivity
    results['specificity'] = specificity

    # Save looped variables
    # variables['path_libri'] = path_libri
    # variables['path_sentiment'] = path_sentiment
    # variables['path_languages'] = path_languages

    with open(save_path + '%s_%i_results.json' % (language, int(scores[1] * 100)), "w") as f:
        json.dump(results, f)

    # Store parameters to json too
    with open(save_path + '%s%i_parameters.json' % (language, int(scores[1] * 100)), "w") as f:
        json.dump(variables, f)

    return None


if __name__ == '__main__':
    main()
