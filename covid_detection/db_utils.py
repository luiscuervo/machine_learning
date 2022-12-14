import os
import sys
import scipy.io.wavfile
import python_speech_features
import numpy as np
import pandas as pd
import random


# from shutil import copyfile
from pydub import AudioSegment

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Change this to the location of the database directories
# DB_DIR = os.path.dirname(os.path.realpath(__file__)) # + "/../databases"
DB_DIR = '/store/datasets/covid/audiosl'

# Import databases
sys.path.insert(1, DB_DIR)

def save_csv(variable, name, dst):
    import csv
    with open(dst + name, 'w') as file:
        write = csv.writer(file)
        write.writerow(variable)

def write_headers_1(csv_file, header_list, row_name):
    df = pd.read_csv(csv_file, names=row_name)  # header=None?
    df.index = header_list
    df.to_csv('%s' % csv_file)
    return None


def write_headers(csv_file, header_list):
    df = pd.read_csv(csv_file, names=header_list)  # header=None?
    df.index = header_list
    df.to_csv('%s' % csv_file)
    return None


def write_column(csv_file, column_list):
    df = pd.read_csv(csv_file)
    df.insert(loc=0, column='', value=column_list)
    df.to_csv('%s' % csv_file, index=False)

    return None


def import_files(folder):
    """Import file contents from IMDb folders."""
    file_names = os.listdir(folder)
    dataset = []
    for file in file_names:
        file = folder + file
        dataset.append(file)
    return dataset


def white_noise(l, intensity):
    noise = np.random.normal(0, 1, l)

    return noise/max(noise) * intensity


def poissonw_noise(img, weight=1):
    # noise_mask = (np.random.poisson(np.abs(img * 255.0 * weight))/weight/255).astype(float)
    noise_mask = (np.random.poisson(np.abs(img))).astype(float)

    return noise_mask


def add_silence_to_dataset(X, files_used, factor=0.1):
    """This funciton will substitute factor% of the samples in X with silence samples"""
    l = len(X)

    z = np.zeros(X.shape)
    X = np.concatenate((X[int(l * factor):], z[:int(l * factor)]), axis=0)
    silences = ["silence" for f in range(int(l * factor))]
    files_used = files_used[int(l * factor):] + silences
    # print(files_used)
    return X, files_used


def add_librispeech_to_dataset(X, files_used, load_type, ratio=0.5, cepstrum_dimension=100, time = 5):
    import joblib
    import csv
    from pydub import AudioSegment

    l = len(X)
    if load_type == "job":
        libri = joblib.load("/store/datasets/jobs/mean_filters_0/librispeech/Xp_librispeech.job")
        with open("/store/datasets/jobs/mean_filters_0/librispeech/librispeech_positives.csv", 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            files_libri = list(reader)[0]

    elif load_type == "import":
        folder = '/store/datasets/aux/all_librispeech_links/'
        files = os.listdir(folder)[:int(l*ratio)]

        audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

        libri = np.zeros((len(files), X.shape[1], X.shape[2]))

        files_libri = []
        c = 0
        fails = 0
        for file in files:
            try:
                sound_file = AudioSegment.from_wav(folder + file)
                audio = sound_file.get_array_of_samples()
                fs = sound_file.frame_rate
                audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
                # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
                libri[c] = audios[0]
                c += 1

                files_libri.append(file)

            except Exception:
                print("Failed to load:", file)
                files.append(os.listdir(folder)[int(l*ratio) + fails])
                fails += 1

    X = np.concatenate((X[int(l * ratio):], libri[:int(l * ratio)]), axis=0)
    files_used = files_used[int(l * ratio):] + files_libri[:int(l * ratio)]

    return X, files_used


def save_to_job(X, Y, dest, file_name):
    """Saves X, Y dataset to jobfile"""
    import joblib
    # time = 1
    # X, Y = get_librispeech_wakeword(time=time)

    # X, Y = np.zeros((1000,500,50)), np.ones(1000)
    joblib.dump((X, Y), dest + file_name)

    # X, Y = joblib.load(DIR_PATH + '/LibriSpeech/XY_7s.job')
    # print(np.shape(X), np.shape(Y))
    return None


def read_job(file):
    import joblib
    X, Y = joblib.load(file)
    return X, Y


def name_cleaner(folder):
    '''Removes certain words from file names in a certain folder'''
    files = os.listdir(folder)

    for f in files:
        n = f.replace("_cough", "")
        n = n.replace("-heavy", "")
        n = n.replace("-shallow", "")
        os.rename(folder + f, DIR_PATH + folder + n)

    return None


def plot_audio(samples, fs):
    import matplotlib.pyplot as plt
    time = np.arange(0, len(samples)) / fs

    plt.plot(time, samples)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time-domain signal')
    plt.show()

    return None

def plot_mfcc(matrix, transpose=True):
    import matplotlib.pyplot as plt
    import datetime
    # fig = plt.figure()
    if transpose:
        matrix = np.transpose(matrix)
    plt.imshow(matrix, interpolation="nearest", origin="upper")
    plt.colorbar()
    plt.show()

    return None

def plot_sound_and_matrix(samples, fs, matrix, name='sound and matrix'):
    import matplotlib.pyplot as plt
    time = np.linspace(0, len(samples)/fs, 1/fs)
    fig, axs = plt.subplots(2)
    fig.suptitle(name)
    axs[0].plot(time, samples)
    axs[1].imshow(np.transpose(matrix), interpolation="nearest", origin="upper")

    plt.show()

def plot_mfcc_and_save(matrix, name='mfcc'):
    import matplotlib.pyplot as plt
    import datetime
    fig = plt.figure()
    plt.imshow(matrix, interpolation="nearest", origin="upper")
    plt.colorbar()
    time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")

    if "plots" not in os.listdir("/store/experiments/covid/"):
        os.mkdir("/store/experiments/covid/plots/")

    fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))


def plot_and_save(data, fs, file_name="None"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import datetime
    time = np.arange(0, len(data)) / fs
    fig = plt.figure()
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time-domain signal')

    if "plots" not in os.listdir("/store/experiments/covid/"):
        os.mkdir("/store/experiments/covid/plots/")

    if file_name != "None":
        name = file_name
    else:
        name = "plot"
    time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")
    fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))
    print("figure saved to: ", '/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))

    # plt.show()

    return None

########### DATASET FUNCTIONS #####################

def delete_files_from_job(X, files_used: list, list_path: str):
    """Locates files from list l in the files_used list, and deletes the indexes where there is a match"""
    import csv
    print('original number of files: ', len(files_used))

    with open(list_path, 'r') as f:
        reader = csv.reader(f)
        l = list(reader)[0]

    indexes = [files_used.index(f) for f in l if f in files_used]
    X = np.delete(X, indexes, axis=0)
    files_used = [f for f in files_used if f not in l]

    print('new number of files: ', len(files_used))

    return X, files_used


def combine_datasets(X_pos1, X_neg1, files_p1, files_n1, X_pos2, X_neg2, files_p2, files_n2):
    """Combine two datasets, supposing that each language is already balanced"""

    X_pos = np.concatenate((X_pos1, X_pos2))
    X_neg = np.concatenate((X_neg1, X_neg2))

    files_p = files_p1 + files_p2
    files_n = files_n1 + files_n2

    return X_pos, X_neg, files_p, files_n


def combine_datasets_balanced(X_pos1, X_neg1, files_p1, files_n1, X_pos2, X_neg2, files_p2, files_n2):
    """Combine two datasets in a balanced way, supposing that each language is already balanced"""

    lengths = [len(X_pos1), len(X_pos2)]
    m = min(lengths)

    if lengths[lengths == m] == 1:
        X_pos1 = X_pos1[:m]
        files_p1 = files_p1[:m]

        X_neg1 = X_neg1[:m]
        files_n1 = files_n1[:m]
    else:
        X_pos2 = X_pos2[:m]
        files_p2 = files_p2[:m]

        X_neg2 = X_neg2[:m]
        files_n2 = files_n2[:m]

    X_pos = np.concatenate((X_pos1, X_pos2))
    X_neg = np.concatenate((X_neg1, X_neg2))

    files_p = files_p1 + files_p2
    files_n = files_n1 + files_n2

    return X_pos, X_neg, files_p, files_n


def reshape_TimeDistributed(X, windows=5):
    X = X.reshape(X.shape[0], windows, int(X.shape[1] / windows), X.shape[2], X.shape[3])

    return X


def normalize_dataset(X):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    print('Normalizing: mean=', mean, 'std=', std)

    return X

###################### WORKING WITH CHUNKS #####################################

def remove_silences(X, files, folder, chunks, threshold=1000, max_t=5):
    """this function will take the dataset split into windows, and delete silences"""
    from pydub import AudioSegment
    count = 0
    for file in files:
        sound_file = AudioSegment.from_wav(folder + file)
        audio = sound_file.get_array_of_samples()
        fs = sound_file.frame_rate

        max_len = max_t * fs
        # plot_audio(samples, fs)

        l = len(audio)
        if max_len < l:
            samples = audio[:max_len]
            samples = np.asarray(samples)
        elif l < max_len:
            d = max_len - l
            samples = np.concatenate((np.zeros(d), np.asarray(audio)))

        temp = 0
        for i in np.arange(int(len(audio)/chunks), len(audio), int(len(audio)/chunks)):
            #if not np.any(np.abs(audio[j]) > 10000 for j in audio[temp:i]):
            if np.mean(np.abs(audio[temp:i])) < threshold:
                X = np.delete(X, count, 0)
                print('silence chunk deleted', print(np.mean(np.abs(audio[temp:i]))))
            else:
                count += 1

            temp = i

    return X

def chunk_maker_random(X, windows):
    """Will split each file in the dataset into many files (N=windows) with chunks inserted into a zero array at
    random position"""
    cepstrum = X.shape[2]
    original_len = X.shape[1]
    win_len = int(X.shape[1] / windows)
    X_new = np.zeros((X.shape[0] * windows, X.shape[1], X.shape[2]))
    X = X.reshape(X.shape[0], windows, win_len, cepstrum)

    # print("shape X_new:", X_new.shape)

    count = 0
    for i in X:
        for j in i:
            X_0 = np.zeros((original_len - win_len, cepstrum))
            rand = random.randint(0, original_len - win_len + 1)
            if rand == original_len - win_len + 1:  # The insert function does not append at the last position
                X_new[count] = np.concatenate((X_0, j), axis=0)
                count += 1
            else:
                X_new[count] = np.insert(X_0, rand, j, axis=0)
                count += 1

    return X_new

def chunk_maker_random_y(X, y, windows):
    """Will split each file in the dataset into many files (N=windows) with chunks inserted into a zero array at
    random position """
    print('train: X ', len(X), 'y ', len(y))

    cepstrum = X.shape[2]
    original_len = X.shape[1]
    win_len = int(X.shape[1] / windows)
    X_new = np.zeros((X.shape[0] * windows, X.shape[1], X.shape[2]))
    X = X.reshape(X.shape[0], windows, win_len, cepstrum)

    y_new = np.zeros(len(y)*windows)
    # print("shape X_new:", X_new.shape)

    count = 0
    count_2 = 0
    for i in X:
        for rep in range(windows):
            y_new[count_2*windows + rep] = y[count_2]

        for j in i:
            X_0 = np.zeros((original_len - win_len, cepstrum))
            rand = random.randint(0, original_len - win_len + 1)
            if rand == original_len - win_len + 1:  # The insert function does not append at the last position
                X_new[count] = np.concatenate((X_0, j), axis=0)
                count += 1
            else:
                X_new[count] = np.insert(X_0, rand, j, axis=0)
                count += 1

        count_2 += 1

    return X_new, y_new


def combine_3_chunks(X_p, X_n):
    X_p1 = chunk_maker_random(X_p[0:int(len(X_p) / 3)], windows=5)
    X_p2 = combine_two_chunks_random(X_p[int(len(X_p) / 3):2 * int(len(X_p) / 3)], windows=5, N=2, normalized=False)
    X_p3 = combine_two_chunks_random(X_p[2 * int(len(X_p) / 3):len(X_p)], windows=5, N=3, normalized=False)
    X_pos = np.concatenate((X_p1, np.concatenate((X_p2, X_p3))))

    X_n1 = chunk_maker_random(X_n[0:int(len(X_n) / 3)], windows=5)
    X_n2 = combine_two_chunks_random(X_n[int(len(X_n) / 3):2 * int(len(X_n) / 3)], windows=5, N=2, normalized=False)
    X_n3 = combine_two_chunks_random(X_n[2 * int(len(X_n) / 3):len(X_n)], windows=5, N=3, normalized=False)
    X_neg = np.concatenate((X_n1, np.concatenate((X_n2, X_n3))))

    return X_pos, X_neg


def combine_two_chunks_random(X, windows, N=2, normalized=False):
    cepstrum = X.shape[2]
    original_len = X.shape[1]
    win_len = int(X.shape[1] / windows)

    X_new = []
    if normalized:
        X = X.reshape(X.shape[0], windows, win_len, cepstrum, X.shape[3])
    else:
        X = X.reshape(X.shape[0], windows, win_len, cepstrum)

    if N == 1:
        X_new = chunk_maker_random(X, windows)

    if N == 2:
        for i in X:
            for j in range(len(i) - N):
                if normalized:
                    X_0 = np.zeros((original_len - N * win_len, cepstrum, 1))
                else:
                    X_0 = np.zeros((original_len - N * win_len, cepstrum))

                A = np.concatenate((i[j], i[j + 1]))
                rand = random.randint(0, original_len - N * win_len + 1)
                if rand == original_len - N * win_len + 1:
                    X_new.append(np.concatenate((X_0, A), axis=0))
                else:
                    X_new.append(np.insert(X_0, rand, A, axis=0))

    if N == 3:
        for i in X:
            for j in range(len(i) - N):
                if normalized:
                    X_0 = np.zeros((original_len - N * win_len, cepstrum, 1))
                else:
                    X_0 = np.zeros((original_len - N * win_len, cepstrum))

                A = np.concatenate((i[j], np.concatenate((i[j + 1], i[j + 2]))))
                rand = random.randint(0, original_len - N * win_len + 1)
                if rand == original_len - N * win_len + 1:
                    X_new.append(np.concatenate((X_0, A), axis=0))
                else:
                    X_new.append(np.insert(X_0, rand, A, axis=0))

    X_new = np.asarray(X_new)

    return X_new


def mfcc_stats(mfcc):
    print('mean = ', np.mean(mfcc), 'std = ', np.std(mfcc))
    print('max value = ', np.max(mfcc), 'min value = ', np.min(mfcc))


def chunk_maker(X, windows, normalized=True):
    cepstrum = X.shape[2]
    original_len = X.shape[1]
    win_len = int(X.shape[1] / windows)

    if normalized:
        X_new = np.zeros((X.shape[0], windows, X.shape[1], X.shape[2], X.shape[3]))
        X = X.reshape(X.shape[0], windows, win_len, cepstrum, X.shape[3])
    else:
        X_new = np.zeros((X.shape[0], windows, X.shape[1], X.shape[2]))
        X = X.reshape(X.shape[0], windows, win_len, cepstrum)

    # print("shape X_new:", X_new.shape)

    count_i = 0
    for i in X:
        count_j = 0
        for j in i:
            if normalized:
                X_0 = np.zeros((original_len - win_len, cepstrum, 1))
            else:
                X_0 = np.zeros((original_len - win_len, cepstrum))

            if count_j == windows - 1:
                X_new[count_i, count_j] = np.concatenate((X_0, j), axis=0)
                count_j += 1
                continue

            X_new[count_i, count_j] = np.insert(X_0, count_j * win_len, j, axis=0)
            count_j += 1

        count_i += 1

    return X_new


################################################################################
###################### IMPORT LANGUAGES + BALANCE DATA #########################
################################################################################

def mfcc_maker(files, time, cepstrum_dimension, resample=False):
    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X = []
    fails = []
    for file in files:
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate

            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    return X, fails


def arrange_dataset(X_pos, X_neg, files_p, files_n, state, test_size=0.2, val_split=0.2, reshaped=True, insert=True, pre_shuffle=True):
    """Returns training, validation, and testing dataset with labeling in a [0,1,0,1,0,1,...] fashion. Every set of data will be balanced"""
    from sklearn.utils import shuffle

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

    if reshaped:
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
    else:
        # We will set the state element as the index 0 element
        X_pos = np.concatenate((X_pos[state:, :, :], X_pos[:state, :, :]), axis=0)
        X_neg = np.concatenate((X_neg[state:, :, :], X_neg[:state, :, :]), axis=0)

        files_p = files_p[state:] + files_p[:state]
        files_n = files_n[state:] + files_n[:state]

        # TRAIN AND TEST SPLIT
        target_test = int(l * test_size)
        target_val = int(l * val_split)
        # positives:
        X_test_p = X_pos[0:target_test, :, :]
        files_test_p = files_p[0:target_test]
        X_val_p = X_pos[target_test:target_test + target_val, :, :]
        files_val_p = files_p[target_test:target_test + target_val]
        X_train_p = X_pos[target_test + target_val:, :, :]
        files_train_p = files_p[target_test + target_val:]

        # negatives
        X_test_n = X_neg[0:target_test, :, :]
        files_test_n = files_n[0:target_test]
        X_val_n = X_neg[target_test:target_test + target_val, :, :]
        files_val_n = files_n[target_test:target_test + target_val]
        X_train_n = X_neg[target_test + target_val:, :, :]
        files_train_n = files_n[target_test + target_val:]

    # Mix in order: [0, 1, 0, 1, ...]
    if insert:
        X_train = np.insert(X_train_p, np.arange(len(X_train_n)), X_train_n, axis=0)
        Y_train = np.insert(np.ones(len(X_train_p)), np.arange(len(X_train_n)), np.zeros(len(X_train_n)))

        X_val = np.insert(X_val_p, np.arange(len(X_val_n)), X_val_n, axis=0)
        Y_val = np.insert(np.ones(len(X_val_p)), np.arange(len(X_val_n)), np.zeros(len(X_val_n)))

        X_test = np.insert(X_test_p, np.arange(len(X_test_n)), X_test_n, axis=0)
        Y_test = np.insert(np.ones(len(X_test_p)), np.arange(len(X_test_n)), np.zeros(len(X_test_n)))
    else:
        X_train = np.concatenate((X_train_p, X_train_n), axis=0)
        Y_train = np.concatenate((np.ones(len(X_train_p)), np.zeros(len(X_train_n))))

        X_val = np.concatenate((X_val_p, X_val_n), axis=0)
        Y_val = np.concatenate((np.ones(len(X_val_p)), np.zeros(len(X_val_n))))

        X_test = np.concatenate((X_test_p, X_test_n), axis=0)
        Y_test = np.concatenate((np.ones(len(X_test_p)), np.zeros(len(X_test_n))))

    # Create dictionary with all the files used in each set
    files = {'train positives': files_train_p, 'train negatives': files_train_n, 'test positives': files_test_p,
             'test negatives': files_test_n, 'validation positives': files_val_p, 'validation negatives': files_val_n}

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, files


def arrange_dataset_no_files(X_pos, X_neg, state, test_size=0.2, val_split=0.2):
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
    if state % 2 != 0:
        state -= 1
        print("making state even by substracting 1: state =", state)

    # We will set the state element as the index 0 element
    X_pos = np.concatenate((X_pos[state:, :, :, :], X_pos[:state, :, :, :]), axis=0)
    X_neg = np.concatenate((X_neg[state:, :, :, :], X_neg[:state, :, :, :]), axis=0)

    # TRAIN AND TEST SPLIT
    target_test = int(l * test_size)
    target_val = int(l * val_split)
    # positives:
    X_test_p = X_pos[0:target_test, :, :, :]
    X_val_p = X_pos[target_test:target_test + target_val, :, :, :]
    X_train_p = X_pos[target_test + target_val:, :, :, :]

    # negatives
    X_test_n = X_neg[0:target_test, :, :, :]
    X_val_n = X_neg[target_test:target_test + target_val, :, :, :]
    X_train_n = X_neg[target_test + target_val:, :, :, :]

    # Mix in order: [0, 1, 0, 1, ...]
    X_train = np.insert(X_train_p, np.arange(len(X_train_n)), X_train_n, axis=0)
    Y_train = np.insert(np.ones(len(X_train_p)), np.arange(len(X_train_n)), np.zeros(len(X_train_n)))

    X_val = np.insert(X_val_p, np.arange(len(X_val_n)), X_val_n, axis=0)
    Y_val = np.insert(np.ones(len(X_val_p)), np.arange(len(X_val_n)), np.zeros(len(X_val_n)))

    X_test = np.insert(X_test_p, np.arange(len(X_test_n)), X_test_n, axis=0)
    Y_test = np.insert(np.ones(len(X_test_p)), np.arange(len(X_test_n)), np.zeros(len(X_test_n)))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def normalize_dataset_PN(X_pos, X_neg, load_type="import"):
    """Normalize speech recognition and computer vision datasets."""

    if load_type == "z":
        mean = np.mean(np.concatenate((X_pos, X_neg)))
        std = np.std(np.concatenate((X_pos, X_neg)))
        X_pos = (X_pos - mean) / std
        X_neg = (X_neg - mean) / std
        normalization = {"mean": mean, "std": std}
        print("Normalization: z-score")
    elif load_type == "max":
        Imax = np.amax(np.concatenate((X_pos, X_neg)))
        X_pos = X_pos/Imax
        X_neg = X_neg/Imax
        normalization = {"max_value": Imax}
        print("Normalization: dividing by max value")
    else:
        mean = np.mean(np.concatenate((X_pos, X_neg)))
        std = np.std(np.concatenate((X_pos, X_neg)))
        X_pos = (X_pos - mean) / std
        X_neg = (X_neg - mean) / std
        normalization = {"mean": mean, "std": std}
        print("normalization type not detected, doing z-score")

    return X_pos, X_neg, normalization


def reshape_dataset(X):
    """Reshape dataset for Convolution."""
    # num_pixels = X.shape[1]*X.shape[2]

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1).astype('float32')

    return X


def job_maker(language, save_name, src, dst, pos_neg=False, time=5, cepstrum_dimension=100):
    """Save languages mfccs into job files"""
    import joblib
    import csv

    if not pos_neg:
        files = import_files(src + 'full-' + language + '/')
    else:
        files = import_files(src + 'positives/')
        files.extend(import_files(src + 'negatives/'))

    print('saving %s data' % language)
    print("trying to save %i files" % len(files))

    audio_data = DataFetcher(DB_DIR, cepstrum_dimension)

    X = []
    fails = []
    for file in files:
        try:
            sound_file = AudioSegment.from_wav(file)
            audio = sound_file.get_array_of_samples()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X.append(audios[0])
        except:
            print("Failed to load:", file)
            fails.append(file)

    joblib.dump(X, dst + "X_%s.job" % save_name)
    print("saved %s to job file" % save_name)

    if fails != []:
        print("SOME FILES FAILED, creating failed files files")
        with open('F-Files_%s.csv' % language, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fails)

    return save_name


def import_language_bioM(lang_pos, languages_neg, lang_pos_name, lang_neg_names):
    """Returns Mfcc (X) and label (Y) for languages specified. lang_pos is a str and languages_neg can be a string or a list of strings"""
    import joblib

    DB_DIR = '/store/datasets/'

    # pos = joblib.load(DB_DIR + 'jobs/X_%s_cep200.job' % lang_pos)
    pos = joblib.load('/store/datasets/jobs/X_%s.job' % lang_pos)
    pos_files = os.listdir(DB_DIR + 'covid/audiosl/bioM_languages/full-' + lang_pos_name)
    lp = len(pos)
    print('original number of positives = ', lp)

    if type(languages_neg) is str:  # if comparing with only one language
        # neg = joblib.load(DB_DIR + 'jobs/X_%s_cep200.job' % languages_neg)
        neg = joblib.load('/store/datasets/jobs/X_%s.job' % languages_neg)
        neg_files = os.listdir(DB_DIR + 'covid/audiosl/bioM_languages/full-' + lang_neg_names)

        ln = len(neg)
        print('original number of negs = ', ln)

        # Choose randomly as many negatives as positives
        if lp < ln:
            neg = neg[:lp]
            neg_files = neg_files[:lp]

        if ln < lp:
            pos = pos[:ln]
            pos_files = pos_files[:ln]

    # FIX THIS!
    elif type(languages_neg) is list:  # If comparing with a combination of languages
        n_lang = len(languages_neg)
        r = lp % n_lang

        if r != 0:  # We want residue 0 when dividing lp/n_lang
            pos = pos[r:]
            pos_files = pos_files[r:]
            lp = len(pos)

        neg = []
        for l in languages_neg:  # For each language:
            print(l)
            # lang_mfcc = joblib.load(DB_DIR + 'jobs/X_%s_cep200.job' % l)
            lang_mfcc = joblib.load(DB_DIR + 'jobs/X_%s.job' % l)
            neg_files = os.listdir(
                DB_DIR + 'covid/audiosl/bioM_languages/full-' + lang_neg_names[languages_neg.index(l)])
            ln = len(lang_mfcc)
            print('original amount of files = ', ln)
            if ln < lp / n_lang:  # If short on neg files
                d = int(lp / n_lang) - ln  # We will substract as many positives as necessary to make balanced DS
                # print(d)
                pos = pos[n_lang * d:]
                pos_files = pos_files[n_lang * d:]
                lp = len(pos)
                print('positive files reduced to ', lp)

                if neg == []:
                    neg = lang_mfcc


                else:
                    neg = np.concatenate((neg, lang_mfcc), axis=0)
                    # print(np.shape(neg))

            else:
                d = int(lp / n_lang)
                # print(d)
                lang_mfcc = lang_mfcc[:d]
                neg_files = neg_files[:d]
                ln = len(lang_mfcc)
                print('negative files reduced to ', ln)

                if neg == []:
                    neg = lang_mfcc
                else:
                    neg = np.concatenate((neg, lang_mfcc), axis=0)

    print("FINAL: positives = ", len(pos), "negatives = ", len(neg))
    positive_items = np.ones(len(pos), int)
    negative_items = np.zeros(len(neg), int)

    X = np.concatenate((pos, neg), axis=0)
    Y = np.concatenate((positive_items, negative_items))

    return X, Y, pos_files, neg_files


def import_language_bioM_CATEGORICAL(languages, suffix='cep100'):
    """Returns Mfcc (X) and label (Y) for languages specified. lang_pos is a str and languages_neg a list of strings"""
    import joblib

    DB_DIR = '/store/datasets/'

    files_used = []
    len_files = []
    for language in languages:
        job = joblib.load('/store/datasets/jobs/X_%s_%s.job' % (language, suffix))
        job_files = os.listdir(DB_DIR + 'covid/audiosl/bioM_languages/full-' + language)

        globals()['X_%s' % language] = job
        globals()['files_%s' % language] = job_files

        print(language, ' number of files:', len(job))
        len_files.append(len(job))

    lmin = min(len_files)
    print("minimum amount of files = ", lmin)

    # Labeling
    count = 0
    y_one_hot = []
    Y = []
    X = []
    for language in languages:
        l = len(globals()['X_%s' % language])

        if l > lmin:
            d = l - lmin
            globals()['X_%s' % language] = np.delete(globals()['X_%s' % language], range(d), axis=0)
            globals()['files_%s' % language] = np.delete(globals()['files_%s' % language], range(d), axis=0)
            # globals()['X_%s' % language] = globals()['X_%s' % language][:lmin]
            # globals()['files_%s' % language] = globals()['files_%s' % language][:lmin]
            l = len(globals()['X_%s' % language])

        print(language, len(globals()['X_%s' % language]), len(globals()['files_%s' % language]))

        if count == 0:
            X = globals()['X_%s' % language]
        else:
            X = np.concatenate((X, globals()['X_%s' % language]), axis=0)

        for i in range(l):
            Y.extend([count])

        count += 1
        print(count)

    for y in Y:
        if y == 0:
            y_one_hot.append([1, 0, 0, 0])
        elif y == 1:
            y_one_hot.append([0, 1, 0, 0])
        elif y == 2:
            y_one_hot.append([0, 0, 1, 0])
        elif y == 3:
            y_one_hot.append([0, 0, 0, 1])
        else:
            print(y)

    y_one_hot = np.asarray(y_one_hot)
    print(np.shape(y_one_hot))
    print(np.shape(X))
    # print(files_used)

    return X, y_one_hot, files_used


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


def import_language(language, lmin):
    """Returns Mfcc (X) and label (Y) for all files in a language directory"""
    positives = import_files(DB_DIR + '/languages/' + language + '/positives/')
    negatives = import_files(DB_DIR + '/languages/' + language + '/negatives/')

    positives, negatives, files_used = balance_language_min(positives, negatives, language, lmin)

    positive_items = np.ones(len(positives))
    negative_items = np.zeros(len(negatives))

    Y = np.concatenate((positive_items, negative_items))

    audio_data = DataFetcherSentimentSpeech(DIR_PATH, 50)

    X = []
    for file in positives:
        sound_file = AudioSegment.from_wav(file)
        audio = sound_file.get_array_of_samples()
        audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, 500)
        # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
        X.append(audios[0])

    for file in negatives:
        sound_file = AudioSegment.from_wav(file)
        audio = sound_file.get_array_of_samples()
        audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, 500)
        # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
        X.append(audios[0])

    X = np.asarray(X)

    print('Data shape for language %s :' % language, 'X_shape:', np.shape(X), 'Y_shape:', np.shape(Y))

    return X, Y, files_used


def balance_language(Xpos, Xneg, language):
    Lp = len(Xpos)
    Ln = len(Xneg)

    files_used = []

    if Lp > Ln:
        d = Lp - Ln
        Xpos = Xpos[d:]
        all_positives = os.listdir(DB_DIR + '/languages/' + language + '/positives/')
        files_used.extend(all_positives[d:])
        files_used.extend(os.listdir(DB_DIR + '/languages/' + language + '/negatives/'))

    elif Ln > Lp:
        d = Ln - Lp
        Xneg = Xneg[d:]

        files_used.extend(os.listdir(DB_DIR + '/languages/' + language + '/positives/'))
        all_negatives = os.listdir(DB_DIR + '/languages/' + language + '/negatives/')
        files_used.extend(all_negatives[d:])
    else:
        files_used.extend(os.listdir(DB_DIR + '/languages/' + language + '/positives/'))
        files_used.extend(os.listdir(DB_DIR + '/languages/' + language + '/negatives/'))

    print("language %s balanced data" % language, len(Xneg), len(Xpos))
    return Xpos, Xneg, files_used


def import_language_PN_STFT(language, n_fft=512, time=5, resample=False):
    """Returns STFT (X) and label (Y) for all files in a language directory"""

    import librosa
    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"
    # language = 'Coswara'

    positives = import_files(DB_DIR + '/languages/' + language + '/positives/')
    negatives = import_files(DB_DIR + '/languages/' + language + '/negatives/')

    positives, negatives, files_used_p, files_used_n = balance_language_pn(positives, negatives, language)

    # DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR
    X_pos = []
    X_neg = []
    fails = []
    for file in positives:
        try:
            sound_file = AudioSegment.from_wav(file)
            samples = np.array(sound_file.get_array_of_samples(), float)
            fs = sound_file.frame_rate

            max_len = time * fs
            l = len(samples)
            if max_len < l:
                samples = samples[:max_len]
            elif l < max_len:
                d = max_len - l
                samples = np.concatenate((np.zeros(d), samples))

            stft = librosa.stft(samples, n_fft=n_fft)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_pos.append(np.abs(stft))
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    for file in negatives:
        try:
            sound_file = AudioSegment.from_wav(file)
            samples = np.array(sound_file.get_array_of_samples(), float)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            fs = sound_file.frame_rate
            max_len = time * fs

            l = len(samples)
            if max_len < l:
                samples = samples[:max_len]
            elif l < max_len:
                d = max_len - l
                samples = np.concatenate((np.zeros(d), samples))

            stft = librosa.stft(samples, n_fft=n_fft)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_neg.append(np.abs(stft))
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    # print(type(X_pos), len(X_pos), len(X_pos[0]), len(X_pos[0][0]))
    # print(type(X_neg), len(X_neg), len(X_neg[0]), len(X_neg[0][0]))

    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)

    # SAVE FAIL FILES
    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            write = csv.writer(f)
            write.writerow(fails)
            exit()

    print('Data shape for language %s :' % language, 'X_pos:', X_pos.shape, 'X_neg:', X_neg.shape)

    return X_pos, X_neg, files_used_p, files_used_n


def import_language_from_filenames(language, files_p, files_n, cepstrum_dimension=100, time=5, resample=False):
    """Returns Mfcc (X) and label (Y) for all files in a language directory"""

    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"
    # language = 'Coswara'

    positives = DB_DIR + '/languages/' + language + '/positives/'
    negatives = DB_DIR + '/languages/' + language + '/negatives/'

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X_pos = []
    X_neg = []
    fails = []
    for f in files_p:
        file = positives + str(f)
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_pos.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    for f in files_n:
        file = negatives + f
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            # print(file)
            # PlotAudio(audio, fs)
            # exit()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_neg.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X_neg = np.asarray(X_neg)
    X_pos = np.asarray(X_pos)

    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fails)
            exit()

    print('Data shape for language %s :' % language, 'X_pos:', np.shape(X_pos), 'X_neg:', np.shape(X_neg))

    return X_pos, X_neg


def import_language_from_filenames_single(language, files_p, cepstrum_dimension=100, time=5, resample=False):
    """Returns Mfcc (X) for all files in a language directory"""

    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"
    # language = 'Coswara'
    positives = DB_DIR + '/languages/' + language + '/positives/'

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X_pos = []
    X_neg = []
    fails = []
    for f in files_p:
        if f.find('Actor') != -1:
            file = '/store/datasets/aux/Audio_Speech_Actors_01-24/' + f
        else:
            file = positives + str(f)
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_pos.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X_pos = np.asarray(X_pos)

    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fails)
            exit()

    return X_pos


def import_dataset_from_paths(files, cepstrum_dimension=100, time=5):
    """Imports dataset (mfcc) from files list containing full paths"""
    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)

    X_pos = []
    fails = []
    for file in files:
        try:
            sound_file = AudioSegment.from_wav(file)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            # print(file)
            # PlotAudio(audio, fs)
            # exit()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_pos.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X_pos = np.asarray(X_pos)

    # SAVE FAIL FILES
    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(fails)
            print("SOME FILES WERE NOT LOADED! Saved Fails to working directory")

    print('Data shape for language :', 'X:', np.shape(X_pos))

    return X_pos


def import_language_PN(language, cepstrum_dimension=100, time=5, resample=False):
    """Returns Mfcc (X) and label (Y) for all files in a language directory"""

    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"
    # language = 'Coswara'

    positives = import_files(DB_DIR + '/languages/' + language + '/positives/')
    negatives = import_files(DB_DIR + '/languages/' + language + '/negatives/')

    positives, negatives, files_used_p, files_used_n = balance_language_pn(positives, negatives, language)

    # positive_items = np.ones(len(positives))
    # negative_items = np.zeros(len(negatives))

    # Y = np.concatenate((positive_items, negative_items))

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X_pos = []
    X_neg = []
    fails = []
    for file in positives:
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            # print(file)
            # PlotAudio(audio, fs)
            # exit()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_pos.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    for file in negatives:
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            # print(file)
            # PlotAudio(audio, fs)
            # exit()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_neg.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X_pos = np.asarray(X_pos)
    X_neg = np.asarray(X_neg)

    # SAVE FAIL FILES
    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            write = csv.writer(f)
            write.writerow(fails)
            exit()

    print('Data shape for language %s :' % language, 'X_pos:', np.shape(X_pos), 'X_neg:', np.shape(X_neg))

    return X_pos, X_neg, files_used_p, files_used_n

def import_language_PN_one_filter(language, sigma=0.5, Fs=4, cepstrum_dimension=100, time=5, resample=False):
    """Returns Mfcc (X) with an extra row with the result of the filter and label (Y) for all files in a language directory"""

    from apply_filters import apply_mean_filter
    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"
    # language = 'Coswara'

    positives = import_files(DB_DIR + '/languages/' + language + '/positives/')
    negatives = import_files(DB_DIR + '/languages/' + language + '/negatives/')

    positives, negatives, files_used_p, files_used_n = balance_language_pn(positives, negatives, language)

    # positive_items = np.ones(len(positives))
    # negative_items = np.zeros(len(negatives))

    # Y = np.concatenate((positive_items, negative_items))

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X_pos = []
    X_neg = []
    fails = []
    for file in positives:
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            # MFCC
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)

            # FILTER
            filt, _, _ = apply_mean_filter(np.array(audio), fs, sigma, Fs)

            #print(filt.shape)
            #print(audios.shape)
            X_pos.append(np.insert(audios[0], 0, filt, axis=1))

        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    for file in negatives:
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate
            # MFCC
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)

            # FILTER
            filt, _, _ = apply_mean_filter(np.array(audio), fs, sigma, Fs)

            X_neg.append(np.insert(audios[0], 0, filt, axis=1))
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X_pos = np.asarray(X_pos)
    X_neg = np.asarray(X_neg)

    # SAVE FAIL FILES
    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            write = csv.writer(f)
            write.writerow(fails)
            exit()

    print('Data shape for language %s :' % language, 'X_pos:', np.shape(X_pos), 'X_neg:', np.shape(X_neg))

    return X_pos, X_neg, files_used_p, files_used_n


def import_language_PN_ten_filters(language, cepstrum_dimension=100, time=5, resample=False):
    """Returns Mfcc (X) with an extra row with the result of the filter and label (Y) for all files in a language directory"""

    from apply_filters import apply_mean_filter
    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"
    # language = 'Coswara'

    positives = import_files(DB_DIR + '/languages/' + language + '/positives/')
    negatives = import_files(DB_DIR + '/languages/' + language + '/negatives/')

    positives, negatives, files_used_p, files_used_n = balance_language_pn(positives, negatives, language)

    # positive_items = np.ones(len(positives))
    # negative_items = np.zeros(len(negatives))

    # Y = np.concatenate((positive_items, negative_items))

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X_pos = []
    X_neg = []
    fails = []
    for file in positives:
        #try:
        sound_file = AudioSegment.from_wav(file)
        if resample:
            sound_file = sound_file.set_frame_rate(22050)
        audio = sound_file.get_array_of_samples()
        fs = sound_file.frame_rate
        # MFCC
        audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)

        # FILTER
        ratios = [4, 2]
        sigmas = [0.1, 0.3, 0.5, 1, 2]

        filt = np.zeros((500, len(sigmas)*len(ratios)))
        c=0
        for sigma in sigmas:
            for Fs in ratios:
                filt[:, c], _, _ = apply_mean_filter(np.array(audio), fs, sigma, Fs)
                c += 1

        #print(filt.shape)
        #print(audios[0].shape)
        X_pos.append(np.insert(audios[0], 0, np.transpose(filt), axis=1))
        print(len(X_pos), 'positive files done')

#        except Exception:
 #           print("Failed to load:", file)
  #          fails.append(file)

    for file in negatives:
        try:
            sound_file = AudioSegment.from_wav(file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate

            # MFCC
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)

            # FILTER
            ratios = [4, 2]
            sigmas = [0.1, 0.3, 0.5, 1, 2]

            filt = np.zeros((500, len(sigmas) * len(ratios)))
            c = 0
            for sigma in sigmas:
                for Fs in ratios:
                    filt[:, c], _, _ = apply_mean_filter(np.array(audio), fs, sigma, Fs)
                    c += 1

            X_neg.append(np.insert(audios[0], 0, np.transpose(filt), axis=1))
            print(len(X_neg), 'negative files done')

        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X_pos = np.asarray(X_pos)
    X_neg = np.asarray(X_neg)

    # SAVE FAIL FILES
    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            write = csv.writer(f)
            write.writerow(fails)
            exit()

    print('Data shape for language %s :' % language, 'X_pos:', np.shape(X_pos), 'X_neg:', np.shape(X_neg))

    return X_pos, X_neg, files_used_p, files_used_n


def import_language_PN_white_noise(language, files_used_p, files_used_n, noise_rate=0.01, cepstrum_dimension=100, time=5, resample=False):
    """Returns Mfcc (X) and label (Y) for all files in a language directory"""

    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"
    # language = 'Coswara'

    folder_p = DB_DIR + '/languages/' + language + '/positives/'
    folder_n = DB_DIR + '/languages/' + language + '/negatives/'

    # positives, negatives, files_used_p, files_used_n = balance_language_pn(positives, negatives, language)

    # positive_items = np.ones(len(positives))
    # negative_items = np.zeros(len(negatives))

    # Y = np.concatenate((positive_items, negative_items))

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X_pos = []
    X_neg = []
    fails = []
    for file in files_used_p:
        try:
            sound_file = AudioSegment.from_wav(folder_p + file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate

            # Add white noise to the signal
            noise = white_noise(len(audio), np.amax(audio)*noise_rate)
            audio += noise
            # print(file)
            # PlotAudio(audio, fs)
            # exit()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_pos.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    for file in files_used_n:
        try:
            sound_file = AudioSegment.from_wav(folder_n + file)
            if resample:
                sound_file = sound_file.set_frame_rate(22050)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate

            # Add white noise to the signal
            noise = white_noise(len(audio), np.amax(audio)/100)
            audio += noise
            # print(file)
            # PlotAudio(audio, fs)
            # exit()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X_neg.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X_pos = np.asarray(X_pos)
    X_neg = np.asarray(X_neg)

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))

    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fails)
            exit()

    # print('Data shape for language %s :' % language, 'X_pos:', np.shape(X_pos), 'X_neg:', np.shape(X_neg))

    return X, y


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def balance_language_pn(Xpos, Xneg, language):
    Lp = len(Xpos)
    Ln = len(Xneg)

    files_used_p = []
    files_used_n = []

    # DB_DIR = "/Volumes/TOSHIBA EXT/Projetto Conclusso/databases"

    if Lp > Ln:
        d = Lp - Ln
        Xpos = Xpos[d:]
        all_positives = os.listdir(DB_DIR + '/languages/' + language + '/positives/')
        files_used_p.extend(all_positives[d:])
        files_used_n.extend(os.listdir(DB_DIR + '/languages/' + language + '/negatives/'))

    elif Ln > Lp:
        d = Ln - Lp
        Xneg = Xneg[d:]

        files_used_p.extend(os.listdir(DB_DIR + '/languages/' + language + '/positives/'))
        all_negatives = os.listdir(DB_DIR + '/languages/' + language + '/negatives/')
        files_used_n.extend(all_negatives[d:])
    else:
        files_used_p.extend(os.listdir(DB_DIR + '/languages/' + language + '/positives/'))
        files_used_n.extend(os.listdir(DB_DIR + '/languages/' + language + '/negatives/'))

    print("language %s balanced data" % language, len(Xneg), len(Xpos))
    return Xpos, Xneg, files_used_p, files_used_n


def import_language_FULL(language, cepstrum_dimension=50, time=5):  # Ojo! prueba con fallos
    """Returns Mfcc (X) and label (Y) for all files in a language directory"""

    positives = import_files(DB_DIR + '/languages/' + language + '/positives/')
    negatives = import_files(DB_DIR + '/languages/' + language + '/negatives/')

    positives, negatives, files_used = balance_language(positives, negatives, language)

    positive_items = np.ones(len(positives))
    negative_items = np.zeros(len(negatives))

    Y = np.concatenate((positive_items, negative_items))

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)

    X = []
    fails = []
    for file in positives:
        try:
            sound_file = AudioSegment.from_wav(file)
            audio = sound_file.get_array_of_samples()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    for file in negatives:
        try:
            sound_file = AudioSegment.from_wav(file)
            audio = sound_file.get_array_of_samples()
            audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X.append(audios[0])
        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    X = np.asarray(X)

    if fails != []:
        import csv
        print("SOME FILES FAILED, creting failed files directory")
        with open('Fail_files_%s' % language, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fails)
            # exit()

    print('Data shape for language %s :' % language, 'X_shape:', np.shape(X), 'Y_shape:', np.shape(Y))

    return X, Y, files_used


def balance_language_min(Xpos, Xneg, language, lmin):
    Lp = len(Xpos)
    Ln = len(Xneg)

    files_used = []

    # cut positives
    if Lp > lmin:
        d = Lp - lmin
        Xpos = Xpos[d:]
        short_positives = os.listdir(DB_DIR + '/languages/' + language + '/positives/')
        files_used.extend(short_positives[d:])
    else:
        files_used.extend(os.listdir(DB_DIR + '/languages/' + language + '/positives/'))

    # cut negatives
    if Ln > lmin:
        d = Ln - lmin
        Xneg = Xneg[d:]
        short_negatives = os.listdir(DB_DIR + '/languages/' + language + '/negatives/')
        files_used.extend(short_negatives[d:])
    else:
        files_used.extend(os.listdir(DB_DIR + '/languages/' + language + '/negatives/'))

    print("language %s balanced data" % language, len(Xneg), len(Xpos))
    return Xpos, Xneg, files_used


class DataFetcher:

    def __init__(self, path, cepstrum_dimension,
                 winlen=0.020, winstep=0.01, encoder_filter_frames=8, encoder_stride=5):
        self.path = path
        self.cepstrum_dimension = cepstrum_dimension
        self.winlen = winlen
        self.winstep = winstep
        self.recording = False
        self.frames = np.array([], dtype=np.int16)
        self.samplerate = 16000
        self.encoder_stride = encoder_stride
        self.encoder_filter_frames = encoder_filter_frames

    def get_mcc_from_file(self, f, max_size):

        mfcc_array = []
        counter = 0

        (rate, sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0]) / (rate * max_size / 100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0] // ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n + inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01,
                                                   numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension,
                                                   nfft=2048, lowfreq=0, highfreq=16000 / 2)
                if mfcc.shape[0] < max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension,
                                           nfilt=self.cepstrum_dimension, nfft=2048, lowfreq=0, highfreq=16000 / 2)
        if mfcc.shape[0] < max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array

    def get_mcc_from_audio(self, sig, rate, max_size):

        mfcc_array = []
        counter = 0

        # (rate,sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0]) / (rate * max_size / 100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0] // ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n + inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01,
                                                   numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension,
                                                   nfft=2048, lowfreq=0, highfreq=16000 / 2)
                if mfcc.shape[0] < max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension,
                                           nfilt=self.cepstrum_dimension, nfft=1024, lowfreq=0, highfreq=16000 / 2)
        if mfcc.shape[0] < max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array


################################################################################
######LibriSpeech Dataset for Speech Recognition#####################
################################################################################

class DataFetcherLibriSpeech:

    def __init__(self, path, cepstrum_dimension,
                 winlen=0.020, winstep=0.01, encoder_filter_frames=8, encoder_stride=5):
        self.path = path
        self.cepstrum_dimension = cepstrum_dimension
        self.winlen = winlen
        self.winstep = winstep
        self.recording = False
        self.frames = np.array([], dtype=np.int16)
        self.samplerate = 16000
        self.encoder_stride = encoder_stride
        self.encoder_filter_frames = encoder_filter_frames

    def get_mcc_from_file(self, f, max_size):

        mfcc_array = []
        counter = 0

        (rate, sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0]) / (rate * max_size / 100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0] // ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n + inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01,
                                                   numcep=self.cepstrum_dimension,
                                                   nfilt=self.cepstrum_dimension, nfft=2048, lowfreq=0,
                                                   highfreq=16000 / 2)
                if mfcc.shape[0] < max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension,
                                           nfilt=self.cepstrum_dimension, nfft=3048, lowfreq=0, highfreq=16000 / 2)
        if mfcc.shape[0] < max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array

    def get_mcc_from_audio(self, sig, rate, max_size):

        mfcc_array = []
        counter = 0

        # (rate,sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0]) / (rate * max_size / 100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0] // ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n + inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01,
                                                   numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension,
                                                   nfft=1024, lowfreq=0, highfreq=16000 / 2)
                if mfcc.shape[0] < max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension,
                                           nfilt=self.cepstrum_dimension, nfft=1024, lowfreq=0, highfreq=16000 / 2)
        if mfcc.shape[0] < max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array


def from_flac_file_to_wav(file, src, dst):
    import soundfile as sf

    # files
    # src = "/Users/luisj/Desktop/SentimentAnalysis/dataset-main/raw"
    # dst = "/Users/luisj/Desktop/SentimentAnalysis/dataset-main/WAV"

    a, fs = sf.read(src + '/%s' % file)
    new_filename = file.replace('.flac', '.wav')
    sf.write(dst + '/' + new_filename, a, fs)

    # PlotAudio(a, fs)

    return None


def L_make_librispeech_wavfiles(source_path='/store/datasets/aux'):
    from shutil import copyfile
    src = source_path + '/Librispeech/train-clean-360/'
    dst = source_path + '/Librispeech/train-clean-wav-360/'

    # if 'train-clean-wav-360' not in os.listdir(src + '..'):
    os.mkdir(dst)

    authors = [f for f in os.listdir(src)]
    books = []
    for author in authors:
        # if dst + author not in os.listdir(dst):
        os.mkdir(dst + author)
        books.extend([author + '/' + f for f in os.listdir(src + author + '/')])

    for book in books:
        temp = [f for f in os.listdir(src + book + '/')]
        # new_book = book.replace("train-clean", "train-clean-wav")
        # if dst + book not in os.listdir(dst):
        os.mkdir(dst + book)
        for i in temp:
            if i.find('txt') == -1:
                from_flac_file_to_wav(i, src + book, dst + book)

            if i.find('txt') != -1:
                copyfile(src + book + '/' + i, dst + book + '/' + i)

    return None


def get_librispeech_wakeword(dir_path, time=1, wakeword='THEM', cepstrum_dimension=50):
    """ Arguments:
        Time (in seconds)
        Wakeword (in caps)
        Cepstrum Dimension
    """
    # print(DIR_PATH)
    # (_, trans) = get_librispeech_files(DIR_PATH)
    (audios, trans, _, _) = get_librispeech_wavfiles(dir_path)

    wakeword_audios, non_wakeword_audios = get_wakeword_audios(wakeword, time * 100, cepstrum_dimension, trans, audios)

    Y_pos = np.ones(len(wakeword_audios))
    Y_neg = np.zeros(len(non_wakeword_audios))

    X_pos = np.asarray(wakeword_audios)
    X_neg = np.asarray(non_wakeword_audios)

    print(X_pos.shape)
    X = np.concatenate((np.asarray(X_pos), np.asarray(X_neg)))
    Y = np.concatenate((Y_pos, Y_neg))

    print(len(Y))
    return X, Y


def get_librispeech_wavfiles(source_path):
    authors = [source_path + '/LibriSpeech/train-clean-wav-360/' + f for f in
               os.listdir(source_path + '/LibriSpeech/train-clean-wav-360/')]
    books = []
    for author in authors:
        books.extend([author + '/' + f for f in os.listdir(author + '/')])

    audios_or_trans = []
    x = 0
    for book in books:
        temp = [book + '/' + f for f in os.listdir(book + '/')]

        for i in temp:
            audios_or_trans.append([i])

    audios = []
    trans = []
    i = 0
    for audio_or_trans_group in audios_or_trans:
        for audio_or_trans in audio_or_trans_group:

            if audio_or_trans.find('.wav') != -1:

                audios.append(audio_or_trans)
            else:
                trans.append(audio_or_trans)

    return (audios, trans, authors, books)


def get_wakeword_audios(wakeword, time, cepstrum_dimension, trans, audios, db_dir='/store/datasets/aux'):
    print(2, dir)
    audio_words_pairs = get_words_in_audio(trans)
    # Returns list of pairs audio + corresponding sentence

    wakeword_audios = []
    non_wakeword_audios = []
    wakeword_sentences = []
    audio_data = DataFetcherLibriSpeech(db_dir, cepstrum_dimension)

    audio_count = 0
    progress_counter = 0
    print('Converting to MFCC...')
    for pair in audio_words_pairs:
        if (progress_counter % 10000) == 0:
            print(str(round(progress_counter / len(audio_words_pairs) * 100)) + '% remaining...')
        progress_counter += 1

        sentence, _ = get_x_words_from_sentence(pair[1])
        index = pair[1].find(wakeword)

        if index != -1:
            filename = get_file_from_audioname(db_dir, pair[0])
            if get_audio_length(filename) >= time / 100:
                position_percent = index / len(pair[1])
                audios = audio_data.get_mcc_from_file(filename, time)
                # Select audiochunk
                audio_with_word = audios[round(len(audios) * position_percent) - 1]
                wakeword_audios.append(audio_with_word)

                if audio_count < 10:
                    chunk = AudioSegment.from_wav(filename)
                    chunk_num = round(len(audios) * position_percent)
                    t1 = chunk_num * time * 10
                    t2 = (chunk_num + 1) * time * 10
                    chunk = chunk[t1:t2]
                    chunk.export(
                        DIR_PATH + '/' + wakeword + str(audio_count) + '.wav',
                        format='wav',
                    )
                    audio_count += 1

        elif len(non_wakeword_audios) < len(wakeword_audios):
            filename = get_file_from_audioname(db_dir, pair[0])
            if get_audio_length(filename) >= time / 100:
                audios = audio_data.get_mcc_from_file(filename, time)
                non_wakeword_audios.append(audios[0])
                if audio_count < 10:
                    chunk = AudioSegment.from_wav(filename)
                    chunk_num = round(len(audios) * position_percent)
                    t1 = chunk_num * time * 10
                    t2 = (chunk_num + 1) * time * 10
                    chunk = chunk[t1:t2]
                    chunk.export(
                        DIR_PATH + '/NO_' + wakeword + str(audio_count) + '.wav',
                        format='wav',
                    )

    return wakeword_audios, non_wakeword_audios


def get_x_words_from_sentence(sentence, number_of_words=0):
    chosen_words = []
    i = 0
    while 1:
        index = sentence.find(' ')
        i += 1
        if index == -1:
            chosen_words.append(sentence)
            break
        chosen_words.append(sentence[0:index])
        sentence = sentence[index + 1:]
    if number_of_words == 0:
        return (chosen_words, i)
    return (chosen_words[0:number_of_words], i)


# from collections import Counter
def get_file_from_audioname(dir, name):
    idx = name.find('-')
    author = name[0:idx]
    tmp = name[idx + 1:]
    idx = tmp.find('-')
    book = tmp[0:idx]

    file = dir + '/LibriSpeech/train-clean-wav-360/' + author + '/' + book + '/' + name + '.wav'
    return file


def get_audio_length(audio_path):
    audio = AudioSegment.from_file(audio_path)
    return audio.duration_seconds


def get_words_in_audio(trans):
    audioname_sentence_pairs = []
    for tran in trans:
        with open(tran, 'r') as f:
            for line in f:
                index = line.find(' ')
                audio_name = line[0:index]
                sentence = line[index + 1:-1]
                audioname_sentence_pairs.append([audio_name, sentence])
    return audioname_sentence_pairs


################################################################################
######Sentiment Analysis dataset  #####################
################################################################################
class DataFetcherSentimentSpeech:

    def __init__(self, path, cepstrum_dimension,
                 winlen=0.020, winstep=0.01, encoder_filter_frames=8, encoder_stride=5):
        self.path = path
        self.cepstrum_dimension = cepstrum_dimension
        self.winlen = winlen
        self.winstep = winstep
        self.recording = False
        self.frames = np.array([], dtype=np.int16)
        self.samplerate = 16000
        self.encoder_stride = encoder_stride
        self.encoder_filter_frames = encoder_filter_frames

    def get_mcc_from_file(self, f, max_size):

        mfcc_array = []
        counter = 0

        (rate, sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0]) / (rate * max_size / 100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0] // ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n + inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01,
                                                   numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension,
                                                   nfft=2048, lowfreq=0, highfreq=16000 / 2)
                if mfcc.shape[0] < max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension,
                                           nfilt=self.cepstrum_dimension, nfft=2048, lowfreq=0, highfreq=16000 / 2)
        if mfcc.shape[0] < max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array

    def get_mcc_from_audio(self, sig, rate, max_size):

        mfcc_array = []
        counter = 0

        # (rate,sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0]) / (rate * max_size / 100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0] // ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n + inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01,
                                                   numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension,
                                                   nfft=1024, lowfreq=0, highfreq=16000 / 2)
                if mfcc.shape[0] < max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension,
                                           nfilt=self.cepstrum_dimension, nfft=1024, lowfreq=0, highfreq=16000 / 2)
        if mfcc.shape[0] < max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0), (0, 0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array


def get_sentiment_from_filenames(files, cepstrum_dimension=100, time=5):
    """Return arrays of samples for each file in files folder"""
    from pydub import AudioSegment
    folder = '/store/datasets/aux/Audio_Speech_Actors_01-24/'

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR
    X = []
    for f in files:
        try:
            # file = '/Users/luisj/Downloads/cough5s.wav'
            audio = AudioSegment.from_wav(folder + str(f))
            if audio.channels == 2:
                audio, _ = audio.split_to_mono()
            samples = audio.get_array_of_samples()
            fs = audio.frame_rate

            audios = audio_data.get_mcc_from_audio(np.asarray(samples), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X.append(audios[0])

        except Exception:
            print('skipped file: ', f)
            exit()

    return X

def get_sentiment_files(source_path='/store/datasets/aux'):
    '''Returns the files within the Audio_Speech directories'''
    authors = [f for f in os.listdir(source_path + '/Audio_Speech_Actors_01-24/')]
    print(authors)
    books = []
    Y = []
    for author in authors:
        books.extend([author + '/' + f for f in os.listdir(source_path + '/Audio_Speech_Actors_01-24/' + author + '/')])

    for book in books:
        idx = book.find('-')
        book = book[idx + 1:]

        idx = book.find('-')
        # print(book[idx+1:idx+3])
        Y.append(book[idx + 1:idx + 3])

    return books, Y


def get_sentiment_data(DB_DIR='/store/datasets/aux', cepstrum_dimension=100):
    '''Returns the mfcc of the files (x) and its correspondig sentiment label (y)'''
    files, Y = get_sentiment_files()
    audio_data = DataFetcherSentimentSpeech(DIR_PATH, cepstrum_dimension)

    X = []
    for file in files:
        file = DB_DIR + '/Audio_Speech_Actors_01-24/' + file

        sound_file = AudioSegment.from_wav(file)
        audio = sound_file.get_array_of_samples()
        audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, 500)
        # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
        X.append(audios[0])

    y_one_hot = []
    for y in Y:
        if y == '01':
            y_one_hot.append([1, 0, 0, 0, 0, 0, 0, 0])
        elif y == '02':
            y_one_hot.append([0, 1, 0, 0, 0, 0, 0, 0])
        elif y == '03':
            y_one_hot.append([0, 0, 1, 0, 0, 0, 0, 0])
        elif y == '04':
            y_one_hot.append([0, 0, 0, 1, 0, 0, 0, 0])
        elif y == '05':
            y_one_hot.append([0, 0, 0, 0, 1, 0, 0, 0])
        elif y == '06':
            y_one_hot.append([0, 0, 0, 0, 0, 1, 0, 0])
        elif y == '07':
            y_one_hot.append([0, 0, 0, 0, 0, 0, 1, 0])
        elif y == '08':
            y_one_hot.append([0, 0, 0, 0, 0, 0, 0, 1])  # Originalmente [1,0,0,0,0,0,0,1]
        else:
            print(y)

    """for y in Y:
        if y == '01':
            y_one_hot.append(0)
        elif y == '02':
            y_one_hot.append(0)
        elif y == '03':
            y_one_hot.append(0)
        elif y == '04':
            y_one_hot.append(0)
        elif y == '05':
            y_one_hot.append(1)
        elif y == '06':
            y_one_hot.append(0)
        elif y == '07':
            y_one_hot.append(0)
        elif y == '08':
            y_one_hot.append(0) #Originalmente [1,0,0,0,0,0,0,1]
        else:
            print(y)"""

    X = np.asarray(X)
    y_one_hot = np.asarray(y_one_hot)
    print(len(X))
    print(len(y_one_hot))
    print(len(Y))

    return X, y_one_hot
    # audios_or_trans = []
    # for book in books:
    #     audios_or_trans.append([source_path + '/LibriSpeech/train-clean-360/' + book + '/' + f for f in os.listdir(source_path + '/LibriSpeech/train-clean-360/' + book + '/')])
    #
    # audios = []
    # trans = []
    # i = 0
    # for audio_or_trans_group in audios_or_trans:
    #     for audio_or_trans in audio_or_trans_group:
    #
    #         if (audio_or_trans.find('.flac') != -1):
    #             audios.append(audio_or_trans)
    #         else:
    #             trans.append(audio_or_trans)
    #
    #
    # return (audios, trans)


#################################################################################

def trying_to_add_noise(file = "/Users/luisj/Downloads/frase1.wav"):
    sound_file = AudioSegment.from_wav(file)
    wav = sound_file.get_array_of_samples()
    l = len(wav)
    fs = sound_file.frame_rate
    # print(fs)
    # X[i] = np.concatenate((X[i], X[i], X[i], X[i], X[i]), axis=0)

    print(max(wav))
    noise = np.random.normal(0, 1, l)
    noise = noise / max(noise) * 0.001 * max(wav)

    noisy_file = wav + noise
    #print(noisy_file)
    #plot_audio(noisy_file, fs)

    from scipy.io.wavfile import write
    write('test.wav', fs, noisy_file)

    #plot_audio(np.random.rand(44100)*3, 44100)

def create_stft_datasets(language):
    import joblib
    import csv
    X_pos, X_neg, files_p, files_n = import_language_PN_STFT(language)

    if language not in os.listdir("/store/datasets/jobs/stft/"):
        os.mkdir("/store/datasets/jobs/stft/" + language)

    dst = "/store/datasets/jobs/stft/%s/" % language
    joblib.dump(X_pos, dst + "Xp_%s.job" % language)
    joblib.dump(X_neg, dst + "Xn_%s.job" % language)

    with open(dst + '%s_positives.csv' % language, 'w') as file:
        write = csv.writer(file)
        write.writerow(files_p)

    with open(dst + '%s_negatives.csv' % language, 'w') as file:
        write = csv.writer(file)
        write.writerow(files_n)

def main():
    import joblib
    import csv
    language = "CIC"
    X_pos, X_neg, files_p, files_n = import_language_PN_one_filter(language)

    if "mfcc+1filt" not in os.listdir("/store/datasets/jobs/"):
        os.mkdir("/store/datasets/jobs/mfcc+1filt/")

    if language not in os.listdir("/store/datasets/jobs/mfcc+1filt/"):
        os.mkdir("/store/datasets/jobs/mfcc+1filt/" + language)

    dst = "/store/datasets/jobs/mfcc+1filt/%s/" % language

    joblib.dump(X_pos, dst + "Xp_%s.job" % language)
    joblib.dump(X_neg, dst + "Xn_%s.job" % language)

    with open(dst + '%s_positives.csv' % language, 'w') as file:
        write = csv.writer(file)
        write.writerow(files_p)

    with open(dst + '%s_negatives.csv' % language, 'w') as file:
        write = csv.writer(file)
        write.writerow(files_n)

    return None


if __name__ == '__main__':
    main()

