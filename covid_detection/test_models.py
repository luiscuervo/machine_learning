import os
import sys
# import datetime
import numpy as np
# import pandas as pd
import joblib
import csv
# import pandas as pd
from pydub import AudioSegment

from tensorflow.keras import models

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Change this to the location of the database directories
DB_DIR = DIR_PATH + 'filters/'

# Import databases
sys.path.insert(1, DB_DIR)

# from MIT_models import choose_multimodel4, choose_model, train_model_WO, train_model_full
from db_utils import plot_and_save, import_language_FULL, import_language_PN, reshape_dataset, chunk_maker, \
    chunk_maker_random, combine_two_chunks_random, normalize_dataset, plot_mfcc, add_librispeech_to_dataset, save_csv, \
    import_dataset_from_paths

def import_job_from_filenames(job_path_p, file_names_p, job_path_n, file_names_n):
    X_p = np.asarray(joblib.load(job_path_p))
    X_n = np.asarray(joblib.load(job_path_n))


    # POSITIVES
    path_p = os.path.dirname(job_path_p) + '/'

    csv1 = [f for f in os.listdir(path_p) if f.find('positives.csv') != -1]
    with open(path_p + csv1[0], 'r') as my_file:
        reader = csv.reader(my_file, delimiter=',')
        files_used_p = list(reader)[0]

    index_p = [i for i in range(len(files_used_p)) if files_used_p[i] in file_names_p]
    # print(index_p)

    print('previous shape (positives)', X_p.shape)
    X_p = X_p[np.array(index_p)]
    print('second shape (positives)', X_p.shape)

    # NEGATIVES
    path_n = os.path.dirname(job_path_n) + '/'
    csv2 = [f for f in os.listdir(path_n) if f.find('negatives.csv') != -1]
    with open(path_n + csv2[0], 'r') as my_file:
        reader = csv.reader(my_file, delimiter=',')
        files_used_n = list(reader)[0]

    index_n = [i for i in range(len(files_used_n)) if files_used_n[i] in file_names_n]
    print(index_p)

    print('previous shape (negatives)', X_n.shape)
    X_n = X_n[np.array(index_n)]
    print('second shape (negatives)', X_n.shape)

    return X_p, X_n

def import_job_from_filenames_single(job_path_p, file_names_p):
    X_p = np.asarray(joblib.load(job_path_p))

    # POSITIVES
    path_p = os.path.dirname(job_path_p) + '/'

    csv1 = [f for f in os.listdir(path_p) if f.find('positives.csv') != -1]
    with open(path_p + csv1[0], 'r') as my_file:
        reader = csv.reader(my_file, delimiter=',')
        files_used_p = list(reader)[0]

    index_p = [i for i in range(len(files_used_p)) if files_used_p[i] in file_names_p]
    # print(index_p)

    print('previous shape (single)', X_p.shape)
    X_p = X_p[np.array(index_p)]
    print('second shape (single)', X_p.shape)

    return X_p

def get_mfc_from_wav(file, time=5, cepstrum_dimension=100):
    from db_utils import DataFetcher
    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)

    sound_file = AudioSegment.from_wav(file)
    audio = sound_file.get_array_of_samples()
    fs = sound_file.frame_rate

    audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
    # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)

    X = np.asarray(audios)
    return X


def test_model_with_silence(model, normalizetion_parameters, normalize=False):
    X = np.zeros((1, 500, 100, 1)).astype(float)

    if normalize:
        if "max_value" in normalizetion_parameters:
            X = X / normalizetion_parameters["max_value"]
        elif "mean" in normalizetion_parameters:
            X = (X - normalizetion_parameters['mean']) / normalizetion_parameters['std']
        else:
            print('normalization parameters not found. Skipping normalization')

    result = model.predict(X)
    print(result)


def find_wrong_files(model, X, y, files):
    """Evaluates model on given samples and finds files that were predicted wrong"""
    assert len(X) == len(y)
    prediction = np.round(model.predict(X)).flatten()
    print(np.shape(prediction), "must be equal to: ", y.shape)
    errors = np.where(prediction != y)[0]
    #errors = np.unique(errors)
    # errors = prediction != y
    wrong_predictions = [files[i] for i in errors]

    return wrong_predictions

def evaluate_dataset_libri(model, normalization_parameters):
    libri = joblib.load("/store/datasets/jobs/mean_filters_0/librispeech/Xp_librispeech.job")
    if "max_value" in normalization_parameters:
        libri = np.array(libri) / normalization_parameters["max_value"]
    elif "mean" in normalization_parameters:
        libri = (libri - normalization_parameters['mean']) / normalization_parameters['std']
    else:
        print('normalization parameters not found. Skipping normalization')

    print("EVALUATING LIBRISPEECH WITH LABEL 0")
    model.evaluate(libri, np.zeros(len(libri)))

def test_with_files_path(load_type="job", add_librispeech=True):
    """Load cough detection model and test it with its training set"""
    import json
    with open('parameters.json') as f:
        variables = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = variables['GPUs']

    # Load model
    model_path = variables['trained_model']
    exp_path = os.path.dirname(model_path)
    print('testing model %s' % model_path)

    model = models.load_model(model_path)

    # normalize dataset
    normalization_path = exp_path + "/normalization_parameters.json"
    with open(normalization_path) as f:
        normalization_parameters = json.load(f)

    path_to_files = exp_path + '/files_used.json'
    with open(path_to_files) as f:
        files = json.load(f)

    if load_type == "import":
        X_test_p = import_dataset_from_paths(files['test positives'])
        X_test_n = import_dataset_from_paths(files['test negatives'])

        X_test_p = (X_test_p - normalization_parameters['mean'])/normalization_parameters['std']
        X_test_n = (X_test_n - normalization_parameters['mean'])/normalization_parameters['std']

    elif load_type == "job":
        X_test_p, X_test_n = import_job_from_filenames(variables["job_p"], files['test positives'],
                                                       variables["job_n"], files['test negatives'])

        if add_librispeech:
            X_test_lib = import_job_from_filenames_single("/store/datasets/jobs/mean_filters_0/librispeech/Xp_librispeech.job",
                                                          files['test negatives'])
            X_test_n = np.append(X_test_n, X_test_lib, axis=0)

        X_test_p /= normalization_parameters["max_value"]
        X_test_n /= normalization_parameters["max_value"]

    else:
        X_test_p, X_test_n = (None, None)

    X_test_p = reshape_dataset(X_test_p)
    X_test_n = reshape_dataset(X_test_n)

    print('FULL EVALUATION')
    model.evaluate(np.concatenate((X_test_p, X_test_n)), np.concatenate((np.ones(len(X_test_p)), np.zeros(len(X_test_n)))))
    print("EVALUATE POSITIVES")
    model.evaluate(X_test_p, np.ones(len(X_test_p)))
    print("EVALUATE NEGATIVES")
    model.evaluate(X_test_n, np.zeros(len(X_test_n)))

    wrong_positives = find_wrong_files(model, X_test_p, np.ones(len(X_test_p)), files['test positives'])
    wrong_negatives = find_wrong_files(model, X_test_n, np.zeros(len(X_test_n)), files['test negatives'])

    # print("Sensitivity = ", (len(files['test positives'])-len(wrong_positives))/len(files['test positives']))
    # print("specificity = ", (len(files['test negatives'])-len(wrong_negatives))/len(files['test negatives']))

    print(wrong_negatives)

    wrong_libri = [f for f in wrong_negatives if f.find('all_librispeech_links')!=-1]
    wrong_coughs = [f for f in wrong_negatives if f.find('all_librispeech_links')==-1]

    print('Wrong libri = ', len(wrong_libri), 'Wrong coughs = ', len(wrong_coughs))



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    """Load cough detection model and test it with different chunks"""
    import json
    with open('parameters.json') as f:
        variables = json.load(f)

    # Load model
    model_path = variables['trained_model']
    exp_path = os.path.dirname(model_path)

    model = models.load_model(model_path)

    # normalize dataset
    normalization_path = exp_path + "/normalization_parameters.json"
    with open(normalization_path) as f:
        normalization_parameters = json.load(f)

    # mean = normalization_data['normalization mean']
    # std = normalization_data['normalization std']

    # print('mean = ', mean, 'std = ', std)
    # X_pos = normalize_dataset(X_pos)

    #test_model_with_silence(model, normalization_parameters, False)

    language = variables['language']

    if variables['load_type'] == 'job':

        job_p = variables['job_p']
        job_n = variables['job_n']
        X_pos = np.asarray(joblib.load(job_p))
        X_neg = np.asarray(joblib.load(job_n))

        # FILES USED
        path_p1 = os.path.dirname(job_p) + '/'
        # path_p2 = os.path.dirname(job_p2) + '/'
        path_n = os.path.dirname(job_n) + '/'

        csv1 = [f for f in os.listdir(path_p1) if f.find('positives.csv') != -1]
        with open(path_p1 + csv1[0], 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            files_used_p = list(reader)[0]
            # print(len(files_used_p))

        csv2 = [f for f in os.listdir(path_n) if f.find('negatives.csv') != -1]
        with open(path_n + csv2[0], 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            files_used_n = list(reader)[0]
            # print(len(files_used_n))

    # elif variables['load_type'] == 'multi job':
    #   jobs = variables

    else:
        # Import mfcc
        X_pos, X_neg, files_used_p, files_used_n = import_language_PN(language)

    # print(files_used_n)
    X_libri, files_libri = add_librispeech_to_dataset(X_neg, files_used_n, variables["load_type"], 1)
    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))
    files = files_used_p + files_used_n

    if "max_value" in normalization_parameters:
        X_pos = X_pos / normalization_parameters["max_value"]
        X_neg = X_neg / normalization_parameters["max_value"]
        X_libri = X_libri / normalization_parameters["max_value"]
    elif "mean" in normalization_parameters:
        X = (X - normalization_parameters['mean']) / normalization_parameters['std']
    else:
        print('normalization parameters not found. Skipping normalization')

    print("EVALUATE POSITIVES")
    model.evaluate(X_pos, np.ones(len(X_pos)))
    print("EVALUATE NEGATIVES")
    model.evaluate(X_neg, np.zeros(len(X_neg)))
    print("EVALUATE LIBRISPEECH")
    model.evaluate(X_libri, np.zeros(len(X_libri)))
    exit()

    print('X_pos: ', X_pos.shape)
    print('X_neg: ', X_neg.shape)

    files_wrong = find_wrong_files(model, X, y, files)

    print("number of wrong files =", len(files_wrong), "out of: ", len(files))
    print(files_wrong)

    model_name = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    save_csv(files_wrong, "false_predictions_%s.csv" % model_name, model_dir)

    # X_pos = reshape_dataset(X_pos)
    # X_test = combine_two_chunks_random(X_pos, 5, N=1, normalized=True)
    # X_test = chunk_maker(X_pos, 2)
    # print('X_pos: ', X_pos.shape, 'X_test', X_test.shape)
    # print(X_test[0, :, :])
    # print(X_test[1, :, :])
    # X_test = X_test[0:1]

    # evaluate sample
    # print('shapes', X_pos.shape, X_test.shape)
    # print("prediction:", model.predict([X_pos]))
    # print("prediction:", model.predict(X_test[0, 0:2, :, :, :]))

    # for i in (X_test[0,0:5,:,:,:]):
    #   print("mean value: ", np.mean(i))

    # print("prediction:", model.predict(np.full((1,500,100,1),0)))
    # examine audio
    # test_model_with_silence(model, mean, std)
    """sound_file = AudioSegment.from_wav(file)
    audio = sound_file.get_array_of_samples()
    fs = sound_file.frame_rate

    file_name = os.path.basename(file)
    file_name = file_name.replace('.wav', '')
    plot_and_save(audio, fs, file_name)"""

    return None


if __name__ == '__main__':
    main()
