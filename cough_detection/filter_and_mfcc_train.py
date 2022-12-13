import os
import sys
# import datetime
import numpy as np
# import pandas as pd
import joblib
import pandas as pd
import csv
import json

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
    normalize_dataset_PN, job_maker, chunk_maker_random, combine_3_chunks, import_language_from_filenames, \
    add_silence_to_dataset, delete_files_from_job, import_language_PN_ten_filters, import_language_PN_one_filter


def normalize_dataset(X):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    print('Normalizing: mean=', mean, 'std=', std)

    return X


def balance_dataset(X_p, X_n, files_p, files_n):
    lp = len(X_p)
    ln = len(X_n)

    if lp < ln:
        X_n = X_n[:lp]
        files_n = files_n[:lp]
        print('too many negative files. New max = ', lp)
    elif ln < lp:
        X_p = X_p[:ln]
        files_p = files_p[:ln]
        print('too many positive files. New max = ', ln)

    X_p = np.asarray(X_p)
    X_n = np.asarray(X_n)

    return X_p, X_n, files_p, files_n


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


def arrange_dataset(X_pos, X_neg, files_p, files_n, state, test_size=0.2, val_split=0.2):
    """Returns training, validation, and testing dataset with labeling in a [0,1,0,1,0,1,...] fashion. Every set of data will be balanced"""

    state = int(state)

    assert len(X_pos) == len(X_neg)
    l = len(X_pos)
    # X = np.insert(X_pos, np.arange(len(X_neg)), X_neg, axis=0)
    # Y = np.insert(Y_pos, np.arange(len(Y_neg)), Y_neg)

    # Apply
    while state > l:
        state = int(state / 2)
        print("state bigger than dataset, reduced in half: state = ", state)

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
    files['vaildation negatives'] = files_val_n

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, files


def concatenate_filter_and_mfcc(multi_input=False):
    # import variables from parameters.json file
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
    patience = variables[
        'patience']  # Epochs waited without improvements in the monitored value before early stopping the training
    monitor = variables['monitor']  # Monitored value
    lr = variables['lr']  # Learning rate
    opt = variables['opt']
    steps = variables['steps']  # Batch size

    # Apply exp decay if desired
    # exp_decay = variables['exp_decay']
    # if exp_decay:
    #    lr = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=10000, decay_rate=0.95, staircase=True)

    # Import dataset
    # 'asymptomatic-official-web-cat-eng-spa' 'main'
    language = variables['language']
    print('working wih dataset: ', language)
    #language = 'main'

    job1 = variables['job_p']
    job2 = variables['job_n']

    # PRIMERO LEER FILES_USED, LUEGO IMPORT LANGUAGE
    X_pos_f = joblib.load(job1)
    X_neg_f = joblib.load(job2)

    path_p = os.path.dirname(job1) + '/'
    path_n = os.path.dirname(job2) + '/'

    csv1 = [f for f in os.listdir(path_p) if f.find('positives.csv') != -1]
    with open(path_p + csv1[0], 'r') as my_file:
        reader = csv.reader(my_file, delimiter=',')
        files_used_p = list(reader)[0]

    csv2 = [f for f in os.listdir(path_n) if f.find('negatives.csv') != -1]
    with open(path_n + csv2[0], 'r') as my_file:
        reader = csv.reader(my_file, delimiter=',')
        files_used_n = list(reader)[0]

    # Delete files under a certain intensity and size threshold
    defective_positives = '/store/datasets/covid/audiosl/languages/%s/defective_files_p.csv' % language
    defective_negatives = '/store/datasets/covid/audiosl/languages/%s/defective_files_n.csv' % language
    X_pos_f, files_used_p = delete_files_from_job(X_pos_f, files_used_p, defective_positives)
    X_neg_f, files_used_n = delete_files_from_job(X_neg_f, files_used_n, defective_negatives)

    X_pos_f, X_neg_f, files_used_p, files_used_n = balance_dataset(X_pos_f, X_neg_f, files_used_p, files_used_n)
    X_pos_f, X_neg_f, normalization = normalize_dataset_PN(X_pos_f, X_neg_f, 'max I')
    #print(files_used_p)

    #print('dataset:', language)
    X_pos_m, X_neg_m = import_language_from_filenames(language, files_used_p, files_used_n, 100)
    X_pos_m, X_neg_m, normalization_m = normalize_dataset_PN(X_pos_m, X_neg_m, "z")

    # We can do this because filters have different type of normalization than mfcc
    normalization.update(normalization_m)

    # Save normalization parameters to variables dictionary
    with open(save_path + 'normalization_parameters.json', "w") as f:
        json.dump(normalization, f)

    #X_pos_m, X_neg_m = balance_dataset(X_pos_m, X_neg_m)
    #print('files positives:', len(files_used_p))
    #print('files negatives:', len(files_used_n))

    X_pos = np.concatenate((X_pos_f, X_pos_m), axis=2)
    X_neg = np.concatenate((X_neg_f, X_neg_m), axis=2)

    print('shape of positive files: ', X_pos.shape)
    print('shape of negative files: ', X_neg.shape)

    # If needed save to JOB
    # dst = '/store/datasets/jobs/'
    # print('saving to job')
    # joblib.dump((X,Y), dst + "XY_%s_5s.job" % language)

    # DIVIDE EACH FILE INTO CHUNKS
    #X_pos = chunk_maker_random(X_pos, 5)
    #X_neg = chunk_maker_random(X_neg, 5)


    # reshape dataset
    X_pos = reshape_dataset(X_pos)
    X_neg = reshape_dataset(X_neg)

    X_neg, files_used_n = add_silence_to_dataset(X_neg, files_used_n, 0.05)

    # TRAINING
    # Train base models
    """mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        path_libri = train_basemodel('Librispeech', save_path, model_type="ResNet50",
                                     monitor='val_loss', epochs=epochs, lr=lr, save_name= "RN50_t5_c100", batch_size=50, patience=10)

        #path_sentiment = train_basemodel('Sentiment', SAVE_DIR + EXP + '/', model_type="ResNet50",
         #                                monitor='val_loss', epochs=epochs, lr=0.001, save_name="RN50-new_t5_c100", batch_size=10, patience=50)

        # language2 can be a list of languages
        # Possible languages : 'english-AW' 'russian' 'catalan' 'spanish'
        path_languages = train_basemodel('Languages', '/store/experiments/covid/new_basemodels/', monitor='val_loss',
                                         epochs=epochs, lr=lr, language1='english-AW', language2='spanish', batch_size=100, patience=10)"""

    # Or use pretrained models
    # path_languages = '/store/models/bio_markers/72CategoricalLanguages_t5_cep100.h5'
    path_libri = variables['path_libri']
    path_sentiment = variables['path_sentiment']
    path_languages = variables['path_languages']
    # Split data
    test_size = variables['test_size']
    val_split = variables['val_split']
    state = variables['state']
    X_train, y_train, X_val, y_val, X_test, y_test, files = arrange_dataset(X_pos, X_neg, files_used_p, files_used_n,
                                                                            state, test_size, val_split)

    # Alternatively:
    # X = np.concatenate((X_pos, X_neg), axis=0)
    # Y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))

    # X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, Y, test_size, val_split)

    # Save files used to json
    with open(save_path + 'files_used.json', "w") as f:
        json.dump(files, f)

    print('saved files_used.json to:', save_path + 'files_used.json')

    # Check that data has the desired shape
    print('training', np.shape(X_train), np.shape(y_train))
    # print(y_train)
    print('testing', np.shape(X_test), np.shape(y_test))
    # print(y_test)
    print('validation', np.shape(X_val), np.shape(y_val))
    # print(y_val)

    # Apply poisson to training dataset
    X_masked = []
    for i in X_train:
        X_masked.append(poissonw_noise(i))

    X_train = np.concatenate((X_train, X_masked), axis=0)
    y_train = np.concatenate((y_train, y_train))

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
        #model = choose_model(X_train[1][1].shape, classes=1)
        #print(model.summary())

        _, model_path = train_model_full_steps(model, X_train, y_train, save_path, language,
                                               (X_val, y_val), monitor, epochs, steps, opt, lr, patience)

        # Test model
        trained_model = models.load_model(model_path)
        scores = trained_model.evaluate(X_test, y_test, verbose=2)

    # Rename model:
    model_path2 = save_path + '%s_%i.h5' % (language, int(scores[1] * 100))
    os.rename(model_path, model_path2)
    os.rename(save_path + "model_history.csv", save_path + '%s_%i_history.h5' % (language, int(scores[1] * 100)))

    print("model saved to: ", model_path2)

    # Save results to json
    sensitivity = format(1 - scores[2] / len(y_test[y_test == 1]), '.2f')
    specificity = format(1 - scores[3] / len(y_test[y_test == 0]), '.2f')
    results = {'model_name': '%i_%s.h5' % (int(scores[1] * 100), language), 'loss': scores[0], 'accuracy': scores[1],
               'sensitivity': sensitivity, 'specificity': specificity}

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


def main(save_to_job=False):
    # import variables from parameters.json file
    with open('parameters.json') as f:
        variables = json.load(f)

    # Establish visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = variables['GPUs']

    # Make experiment dir
    SAVE_DIR = "/store/experiments/covid/"
    EXP = variables['EXP']              # Folder to be created under experiment name in SAVE_DIR
    save_name = variables['save_name']  # Subdirectory to be created in EXP

    if EXP not in os.listdir(SAVE_DIR):
        os.mkdir(SAVE_DIR + EXP)
    if save_name not in os.listdir(SAVE_DIR + EXP):
        os.mkdir(SAVE_DIR + EXP + '/' + save_name)

    # Results will be saved here:
    save_path = SAVE_DIR + EXP + '/' + save_name + '/'

    # Parameters used:
    epochs = variables['epochs']       # Number of repetitions of the training
    patience = variables['patience']   # Epochs waited without improvements in the monitored value before early stopping
    monitor = variables['monitor']     # Monitored value
    lr = variables['lr']               # Learning rate
    opt = variables['opt']             # Optimizer
    steps = variables['steps']         # Batch size

    # Train base models
    """mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        path_libri = train_basemodel('Librispeech', save_path, model_type="ResNet50",
                                     monitor='val_loss', epochs=epochs, lr=lr, save_name= "RN50_t5_c100", batch_size=50, patience=10)

        #path_sentiment = train_basemodel('Sentiment', SAVE_DIR + EXP + '/', model_type="ResNet50",
         #                                monitor='val_loss', epochs=epochs, lr=0.001, save_name="RN50-new_t5_c100", batch_size=10, patience=50)

        # language2 can be a list of languages
        # Possible languages : 'english-AW' 'russian' 'catalan' 'spanish'
        path_languages = train_basemodel('Languages', '/store/experiments/covid/new_basemodels/', monitor='val_loss',
                                         epochs=epochs, lr=lr, language1='english-AW', language2='spanish', batch_size=100, patience=10)"""

    # Or use pretrained models
    # path_languages = '/store/models/bio_markers/72CategoricalLanguages_t5_cep100.h5'
    path_libri = variables['path_libri']
    path_sentiment = variables['path_sentiment']
    path_languages = variables['path_languages']

    # Import dataset
    # 'asymptomatic-official-web-cat-eng-spa' 'main'
    language = variables['language']
    print('working wih dataset: ', language)

    X_pos, X_neg, files_used_p, files_used_n = import_language_PN_one_filter(language)
    # X_pos, X_neg, files_used_p, files_used_n = import_language_PN_ten_filters(language)

    if save_to_job:
        dst = '/store/datasets/jobs/'
        job_save_name = "10filt+mfcc"
        if job_save_name not in os.listdir(dst):
            os.mkdir(dst + job_save_name)

        dst += job_save_name + '/'

        joblib.dump(X_pos, dst + "Xp_%s.job" % language)
        print("saved positives to:", dst + "Xp_%s.job" % language)
        joblib.dump(X_neg, dst + "Xn_%s.job" % language)
        print("saved negatives to:", dst + "Xn_%s.job" % language)

        with open(dst + '%s_positives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_used_p)

        with open(dst + '%s_negatives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_used_n)

    #print('dataset:', language)
    X_pos, X_neg, normalization = normalize_dataset_PN(X_pos, X_neg, "z")

    # Save normalization parameters to variables dictionary
    with open(save_path + 'normalization_parameters.json', "w") as f:
        json.dump(normalization, f)

    #X_pos_m, X_neg_m = balance_dataset(X_pos_m, X_neg_m)
    #print('files positives:', len(files_used_p))
    #print('files negatives:', len(files_used_n))

    print('shape of positive files: ', X_pos.shape)
    print('shape of negative files: ', X_neg.shape)

    # If needed save to JOB
    # dst = '/store/datasets/jobs/'
    # print('saving to job')
    # joblib.dump((X,Y), dst + "XY_%s_5s.job" % language)

    # DIVIDE EACH FILE INTO CHUNKS
    #X_pos = chunk_maker_random(X_pos, 5)
    #X_neg = chunk_maker_random(X_neg, 5)


    # reshape dataset
    X_pos = reshape_dataset(X_pos)
    X_neg = reshape_dataset(X_neg)

    # X_neg, files_used_n = add_silence_to_dataset(X_neg, files_used_n, 0.05)

    # Split data
    test_size = variables['test_size']
    val_split = variables['val_split']
    state = variables['state']
    X_train, y_train, X_val, y_val, X_test, y_test, files = arrange_dataset(X_pos, X_neg, files_used_p, files_used_n,
                                                                            state, test_size, val_split)

    # Alternatively:
    # X = np.concatenate((X_pos, X_neg), axis=0)
    # Y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))

    # X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, Y, test_size, val_split)

    # Save files used to json
    with open(save_path + 'files_used.json', "w") as f:
        json.dump(files, f)

    print('saved files_used.json to:', save_path + 'files_used.json')

    # Check that data has the desired shape
    print('training', np.shape(X_train), np.shape(y_train))
    # print(y_train)
    print('testing', np.shape(X_test), np.shape(y_test))
    # print(y_test)
    print('validation', np.shape(X_val), np.shape(y_val))
    # print(y_val)

    # Apply poisson to training dataset
    X_masked = []
    for i in X_train:
        X_masked.append(poissonw_noise(i))

    X_train = np.concatenate((X_train, X_masked), axis=0)
    y_train = np.concatenate((y_train, y_train))

    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=1)
    X_val, y_val = shuffle(X_val, y_val, random_state=2)
    X_test, y_test = shuffle(X_test, y_test, random_state=3)

    print('data ready:', 'X_train = ', np.shape(X_train), 'Y_train = ', np.shape(y_train))

    multi_input = False

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
        #model = choose_model(X_train[1][1].shape, classes=1)
        #print(model.summary())

        _, model_path = train_model_full_steps(model, X_train, y_train, save_path, language,
                                               (X_val, y_val), monitor, epochs, steps, opt, lr, patience)

        # Test model
        trained_model = models.load_model(model_path)
        scores = trained_model.evaluate(X_test, y_test, verbose=2)

    # Rename model:
    model_path2 = save_path + '%s_%i.h5' % (language, int(scores[1] * 100))
    os.rename(model_path, model_path2)
    os.rename(save_path + "model_history.csv", save_path + '%s_%i_history.h5' % (language, int(scores[1] * 100)))

    print("model saved to: ", model_path2)

    # Save results to json
    sensitivity = format(1 - scores[2] / len(y_test[y_test == 1]), '.2f')
    specificity = format(1 - scores[3] / len(y_test[y_test == 0]), '.2f')
    results = {'model_name': '%i_%s.h5' % (int(scores[1] * 100), language), 'loss': scores[0], 'accuracy': scores[1],
               'sensitivity': sensitivity, 'specificity': specificity}

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
    #A = joblib.load("/store/datasets/jobs/10filt+mfcc/Xp_Coswara.job")
    #from db_utils import plot_mfcc_and_save
    #plot_mfcc_and_save(A[0])

    #concatenate_filter_and_mfcc()
    main(save_to_job=True)
