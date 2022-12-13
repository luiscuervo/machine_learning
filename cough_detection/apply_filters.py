import os
import sys
import numpy as np
from pydub import AudioSegment
import joblib
import csv

from numba import njit

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Change this to the location of the database directories
DB_DIR = '/Volumes/TOSHIBA EXT/Projetto Conclusso/databases/languages/'

# Import databases
sys.path.insert(1, DB_DIR)

from db_utils import plot_audio, get_sentiment_files, plot_mfcc_and_save, plot_mfcc, get_librispeech_wavfiles


@njit
def apply_mean_filter(audio_samples, fs, sigma, Fs, splits_per_second=100, steps=500):
    """Calculates the difference in mean intensity in neighbor windows, and calculates the difference among them.
    This difference is then appended to the resulting array. POSITIVE FILTER"""

    # splits_per_second = fs
    sigma_s = sigma / Fs
    sigma_s = int(sigma_s * fs)         # number of added samples
    sigma = int(sigma * fs)             # number of added samples

    # audio_samples = np.array(audio_samples, dtype=np.float64)
    # We will work with absolute values of intensity normalized to max value 1
    # print(Imax)
    audio_samples = np.abs(audio_samples) / np.amax(audio_samples)

    # We will store our results here
    X = np.zeros(steps)
    X_l = np.zeros(steps)
    X_r = np.zeros(steps)

    # We will divide the audio signal into 500 steps evenly spaced
    count = 0

    for i in np.linspace(0, len(audio_samples), steps+1)[:-1]:
        i = int(i)
        d = len(audio_samples) - i
        # print(sigma)

        # RIGHT FILTER
        if d < sigma:
            # print('shorten filter right', a)
            sigma = d

        sigma_splits = int(splits_per_second * sigma / fs)
        if sigma_splits == 0:
            sigma_splits = 1

        splitted_sigma = np.array_split(audio_samples[i:i + sigma], sigma_splits)
        first_split_mean = np.mean(splitted_sigma[0])
        # splits_s = np.array_split(audio_samples[i + sigma: i + sigma + sigma_s], int(sigma_splits / Fs))

        if first_split_mean <= 0.05 * max(audio_samples):
            X[count] = 0
            count += 1
            continue

        # RIGHT FILTER
        x_r = np.zeros(sigma_splits)
        for j in np.arange(1, sigma_splits):

            filt = first_split_mean - j / sigma_splits * first_split_mean
            dist = (filt - np.mean(splitted_sigma[j]))
            if dist > 0:
                x_r[j] = 1 - dist/filt      # we divide by the height of the filter at j to penalize big distances

            else:                           # THIS IS INTERESTING!
                # x_r[j] = dist/(1 - filt)     # factor 0.9999 to avoid dividing by 0
                x_r[j] = 0

        # LEFT FILTER
        if i < sigma_s:
            #x_l = np.amax(audio_samples[:i])
            x_l = np.mean(audio_samples[:i]) if i != 0 else 0
        else:
            #x_l = np.amax(audio_samples[i-sigma_s:i])
            x_l = np.mean(audio_samples[i-sigma_s:i])

        x = np.mean(x_r) - x_l

        # X[count] = 255 * np.mean(x_r)
        X[count] = 255 * x if x > 0 else 0
        X_r[count] = np.mean(x_r) * 255
        X_l[count] = -x_l * 255
        count += 1

    return X, X_r, X_l

def apply_mean_filter_2(audio_samples, fs, sigma, Fs, splits_per_second=100, steps=500):
    """Calculates the difference in mean intensity in neighbor windows, and calculates the difference among them.
    This difference is then appended to the resulting array. POSITIVE FILTER"""

    # splits_per_second = fs
    sigma_s = sigma / Fs
    sigma_s = int(sigma_s * fs)         # number of added samples
    sigma = int(sigma * fs)             # number of added samples

    # We will work with absolute values of intensity normalized to max value 1
    Imax = max(audio_samples)
    # print(Imax)
    audio_samples = np.abs(audio_samples) / Imax

    # We will store our results here
    X = np.zeros(steps)

    # We will divide the audio signal into 500 steps evenly spaced
    count = 0

    a = 1
    for i in np.linspace(0, len(audio_samples), steps + 1)[:-1]:
        i = int(i)
        d = len(audio_samples) - i
        # print(sigma)

        # RIGHT FILTER
        if d < sigma:
            # print('shorten filter right', a)
            a += 1
            sigma = d

        sigma_splits = int(splits_per_second * sigma / fs)
        if sigma_splits == 0:
            sigma_splits = 1

        splitted_sigma = np.array_split(audio_samples[i:i + sigma], sigma_splits)
        first_split_mean = np.mean(splitted_sigma[0])
        # splits_s = np.array_split(audio_samples[i + sigma: i + sigma + sigma_s], int(sigma_splits / Fs))

        if first_split_mean <= 0.1 * max(audio_samples):
            X[count] = 0
            count += 1
            continue

        x_r = np.zeros(sigma_splits)
        for j in range(sigma_splits):
            filt = first_split_mean - j / sigma_splits * first_split_mean
            dist = (filt - np.mean(splitted_sigma[j]))
            if dist > 0:
                x_r[j] = 1 - dist/filt      # we divide by the height of the filter at j to penalize big distances

            else:                           # THIS IS INTERESTING!
                x_r[j] = dist/(1-filt*0.999)     # factor 0.9999 to avoid dividing by 0
                # x_r[j] = 0

        # LEFT FILTER
        filter_left = np.zeros(sigma_s)
        for j in range(sigma_s):
            filter_left[i] = - j / sigma_s

        if i < sigma_s:
            x_l = np.sum(filter_left[:i] * audio_samples[:i])
        else:
            #x_l = np.amax(audio_samples[i-sigma_s:i])
            x_l = np.sum(filter_left * audio_samples[i-sigma_s:i])

        # x = x_l - x_r

        if np.mean(x_r) > 0: # and x_l < 0.001 * max(audio_samples):
            X[count] = 255 * np.mean(x_r)
            # X[count] = np.mean(x_r)
        else:
            X[count] = 0
        count += 1

    return X

# @njit
def apply_mean_filter_gradient(audio_samples, fs, sigma, Fs, splits_per_second, steps=500):
    """Calculates the difference in mean intensity in neighbor windows, and calculates the difference among them.
    This difference is then appended to the resulting array. POSITIVE FILTER"""

    splits_per_second = fs
    sigma = int(sigma * fs)  # number of added samples
    sigma_s = int(sigma / Fs * fs)  # number of subtracted samples

    # We will work with absolute values of intensity normalized to max value 1
    Imax = max(audio_samples)
    # print(Imax)
    # audio_samples = np.abs(audio_samples) / Imax

    # We will store our results here
    X = np.zeros(steps)

    # We will divide the audio signal into 500 steps evenly spaced
    count = 0
    for i in np.linspace(0, len(audio_samples), steps + 1)[:-1]:
        i = int(i)
        d = len(audio_samples) - i
        print(d)

        # RIGHT FILTER
        if d < sigma:
            print('shorten filter right')
            sigma = d

        sigma_splits = int(splits_per_second * sigma / fs)
        if sigma_splits == 0:
            sigma_splits = 1

        splitted_sigma = np.array_split(audio_samples[i:i + sigma], sigma_splits)
        first_split_mean = np.mean(splitted_sigma[0])
        # splits_s = np.array_split(audio_samples[i + sigma: i + sigma + sigma_s], int(sigma_splits / Fs))

        if first_split_mean <= Imax / 10:
            X[count] = 0
            count += 1
            continue

        x_r = np.zeros(sigma_splits)
        for j in range(sigma_splits):
            # filt = np.max(audio_samples[i: i + sigma]) - j / sigma_splits * np.max(audio_samples[i:i+sigma])
            filt = first_split_mean - j / sigma_splits * first_split_mean
            x_r[j] = (Imax - np.abs(filt - np.mean(splitted_sigma[
                                                       j]))) / Imax  # * (sigma_splits - j) / sigma_splits      # Since samples are normalized, x_r is always < 1

        # LEFT FILTER
        """if i < sigma_s:
            if i == 0:
                sigma_s_splits = 0
            else:
                sigma_s_temp = i
                sigma_s_splits = int(splits_per_second * sigma_s_temp / fs)
                splitted_sigma_s = np.array_split(audio_samples[i - sigma_s_temp:i], sigma_s_splits)
        else:
            sigma_s_temp = sigma_s
            sigma_s_splits = int(splits_per_second * sigma_s_temp / fs)
            splitted_sigma_s = np.array_split(audio_samples[i - sigma_s_temp:i], sigma_s_splits)

        x_l = np.zeros(sigma_s_splits)
        if sigma_s_splits != 0:
            for k in range(sigma_s_splits):
                x_l[k] = np.mean(splitted_sigma_s[k]) # * k/sigma_s_splits    # We give more weight to the last values
        else:
            x_l = np.zeros(1)"""

        X[count] = 255 * np.mean(x_r)  # * (1 - np.mean(x_l))
        count += 1

    return X

def triangle_filter_1_left(sigma_s, fs, h_d):
    """Returns a two rectangle triangle like filter, with a negative and positive triangles one after the other.
    variables: b=base (in seconds), h=height. Indexes: u=up, d=down. fs is sampling frequency"""
    x = np.zeros(int(sigma_s * fs))

    count = 0
    for p in np.linspace(0, sigma_s, int(sigma_s * fs)):
        x[count] = -h_d / sigma_s * p
        count += 1

    return x


def triangle_filter_1_right(sigma, fs, h_u):
    """Returns a two rectangle triangle like filter, with a negative and positive triangles one after the other.
    variables: b=base (in seconds), h=height. Indexes: u=up, d=down. fs is sampling frequency"""
    x = np.zeros(int(sigma * fs))

    count = 0
    for p in np.linspace(0, sigma, int(sigma * fs)):
        x[count] = (h_u - h_u / sigma * p)
        count += 1

    return x

# It is actually the same as triangle_filter_1_right
def triangle_filter_2_left(sigma, fs, h_u):
    """Returns a two rectangle triangle like filter, with a negative and positive triangles one after the other.
    variables: b=base (in seconds), h=height. Indexes: u=up, d=down. fs is sampling frequency"""

    x = np.zeros(int(sigma * fs))

    count = 0
    for p in np.linspace(0, sigma, int(sigma * fs)):
        x[count] = h_u - h_u / sigma * p
        count += 1

    return x


def triangle_filter_2_right(sigma_s, fs, h_d):
    """Returns a two rectangle triangle like filter, with a negative and positive triangles one after the other.
    variables: b=base (in seconds), h=height. Indexes: u=up, d=down. fs is sampling frequency"""

    x = np.zeros(int(sigma_s * fs))

    count = 0
    for p in np.linspace(0, sigma_s, int(sigma_s * fs)):
        x[count] = -h_d + h_d / sigma_s * p
        count += 1

    return x


# @njit
def get_filtered_file_multiply(samples, sample_filter_left, sample_filter_right, size=500, factor=1):
    """Applies the filters provided into the samples """
    # FIRST FILTER
    l = len(samples)
    samples = np.abs(samples)
    Imax = int(np.amax(samples))

    # samples /= Imax

    # full_filter = np.concatenate((sample_filter_left, sample_filter_right))

    Y1 = np.zeros(size)
    count = 0
    for i in np.linspace(0, l, size + 1)[:-1]:
        i = int(i)
        sigma = len(sample_filter_right)
        sigma_s = len(sample_filter_left)
        # print(i)

        d = l - i
        # For the first positions of the filter
        if i < len(sample_filter_left):
            if i == 0:
                #Y1[count] = 255 * np.sum(samples[:len(sample_filter_right)] * sample_filter_right) / np.sum(sample_filter_right)
                Y1[count] = 255 * np.sum(samples[:len(sample_filter_right)] * sample_filter_right) / \
                            np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right)
                count += 1

            else:
                sample_filter_left_temp = sample_filter_left[len(sample_filter_left) - i:]
                sigma_s = len(sample_filter_left_temp)
                # Y1_r = np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right)  / np.sum(sample_filter_right)
                Y1_r = np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right) / \
                       np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right)
                # Y1_l = np.sum(samples[i - len(sample_filter_left_temp):i] * sample_filter_left_temp)  / np.sum(sample_filter_left_temp)
                Y1_l = np.sum(samples[i - len(sample_filter_left_temp):i] * sample_filter_left_temp) / \
                       np.sum(np.full(len(sample_filter_left_temp), Imax * factor) * sample_filter_left_temp)

                Y1[count] = (-Y1_l + Y1_r) * 255  # OJO! IMPORTANCIA DE CADA LADO?
                count += 1

        # For the last positions of the filter
        elif d < len(sample_filter_right):
            if i == l:
                Y1[count] = 0
                continue
            sample_filter_right_temp = sample_filter_right[:-(len(sample_filter_right) - d)]
            sigma = len(sample_filter_right_temp)

            # Y1_r = np.sum(samples[i:i + len(sample_filter_right_temp)] * sample_filter_right_temp) / np.sum(sample_filter_right_temp)
            Y1_r = np.sum(samples[i:i + len(sample_filter_right_temp)] * sample_filter_right_temp) / \
                   np.sum(np.full(len(sample_filter_right_temp), Imax * factor) * sample_filter_right_temp)
            # Y1_l = np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) / np.sum(sample_filter_left)
            Y1_l = np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) / \
                   np.sum(np.full(len(sample_filter_left), Imax * factor) * sample_filter_left)
            Y1[count] = (-Y1_l + Y1_r) * 255
            count += 1

        else:
            # Y1_r = np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right)  / np.sum(sample_filter_right)
            Y1_r = np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right)  / \
                   np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right)
            # Y1_l = np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left)  / np.sum(sample_filter_left)
            Y1_l = np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) /\
                   np.sum(np.full(len(sample_filter_left), Imax * factor) * sample_filter_left)
            # print(Y1_l)
            Y1[count] = (-Y1_l + Y1_r) * 255
            count += 1

    return np.abs(Y1)

#@njit
def get_filtered_file_multiply_reverse(samples, sample_filter_left, sample_filter_right, size=500, factor=1):
    """Applies the filters provided into the samples """
    # FIRST FILTER
    l = len(samples)
    samples = np.abs(samples)
    Imax = int(np.amax(samples))

    Y1 = np.zeros(size)
    count = 0

    for i in np.linspace(0, l, size + 1)[:-1]:
        i = int(i)
        sigma = len(sample_filter_left)
        sigma_s = len(sample_filter_right)
        # print(i)
        d = l - i

        # For the first positions of the filter, cut off filter left
        if i < sigma:
            if i == 0:
                Y1[count] = 0
                count += 1

            else:
                sample_filter_left_temp = sample_filter_left[len(sample_filter_left) - i:]
                sigma_s = len(sample_filter_left_temp)

                Y1_l = np.sum(samples[i - len(sample_filter_left_temp):i] * sample_filter_left_temp) / \
                       np.sum(np.full(len(sample_filter_left_temp), Imax * factor) * sample_filter_left_temp)
                Y1_r = np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right) /\
                       np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right)

                Y1[count] = (-Y1_l + Y1_r) * 255  # OJO! IMPORTANCIA DE CADA LADO?  # OJO! IMPORTANCIA DE CADA LADO?
                count += 1

        # For the last positions of the filter, cut off filter right
        elif d < sigma_s:
            # if i == l:
            #   Y1[count] = 0
            #  continue
            sample_filter_right_temp = sample_filter_right[:-(len(sample_filter_right) - d)]
            sigma = len(sample_filter_right_temp)

            Y1_l = np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) / \
                   np.sum(np.full(len(sample_filter_left), Imax * factor) * sample_filter_left)
            Y1_r = np.sum(samples[i:i + len(sample_filter_right_temp)] * sample_filter_right_temp) /\
                   np.sum(np.full(len(sample_filter_right_temp), Imax * factor) * sample_filter_right_temp)
            Y1[count] = (-Y1_l + Y1_r) * 255  # OJO! IMPORTANCIA DE CADA LADO?
            count += 1

        # Middle positions
        else:
            # print('step', i, 'of', l)
            Y1_l = np.sum(samples[i - sigma:i] * sample_filter_left) / \
                   np.sum(np.full(sigma, Imax * factor) * sample_filter_left)
            Y1_r = np.sum(samples[i:i + sigma_s] * sample_filter_right) /\
                   np.sum(np.full(sigma_s, Imax * factor) * sample_filter_right)
            # print(Y1_r)

            Y1[count] = (-Y1_l + Y1_r) * 255  # OJO! IMPORTANCIA DE CADA LADO?
            count += 1

    return np.abs(Y1)


@njit
def get_filtered_file_centered(samples, sample_filter_left, sample_filter_right, size=500, factor=1):
    """Applies the filters provided into the samples """
    # FIRST FILTER
    l = len(samples)
    samples = np.abs(samples)
    Imax = int(np.amax(samples))

    # full_filter = np.concatenate((sample_filter_left, sample_filter_right))

    Y1 = np.zeros(size)
    count = 0
    for i in np.linspace(0, l, size + 1)[:-1]:
        i = int(i)
        sigma = len(sample_filter_right)
        sigma_s = len(sample_filter_left)
        # print(i)

        d = l - i
        # For the first positions of the filter
        if i < len(sample_filter_left):
            if i == 0:
                Y1[count] = np.sum(samples[:len(sample_filter_right)] * sample_filter_right) / \
                            np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right) * 255
                count += 1

            else:
                sample_filter_left_temp = sample_filter_left[len(sample_filter_left) - i:]
                sigma_s = len(sample_filter_left_temp)
                Y1_r = np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right) / \
                       np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right)
                Y1_l = Y1_r * (1 - np.sum(samples[i - len(sample_filter_left_temp):i] * sample_filter_left_temp) /
                               np.sum(np.full(len(sample_filter_left_temp), Imax * factor) * sample_filter_left_temp))

                Y1[count] = (sigma_s * Y1_l + sigma * Y1_r) / (sigma + sigma_s) * 255  # OJO! IMPORTANCIA DE CADA LADO?
                count += 1

        # For the last positions of the filter
        elif d < len(sample_filter_right):
            if i == l:
                Y1[count] = 0
                continue
            sample_filter_right_temp = sample_filter_right[:-(len(sample_filter_right) - d)]
            sigma = len(sample_filter_right_temp)

            Y1_r = np.sum(samples[i:i + len(sample_filter_right_temp)] * sample_filter_right_temp) / \
                   np.sum(np.full(len(sample_filter_right_temp), Imax * factor) * sample_filter_right_temp)
            Y1_l = Y1_r * (1 - np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) /
                           np.sum(np.full(len(sample_filter_left), Imax * factor) * sample_filter_left))
            Y1[count] = (sigma_s * Y1_l + sigma * Y1_r) / (sigma + sigma_s) * 255
            count += 1

        else:
            Y1_r = np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right) / \
                   np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right)
            Y1_l = Y1_r * (1 - np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) /
                           np.sum(np.full(len(sample_filter_left), Imax * factor) * sample_filter_left))
            Y1[count] = (sigma_s * Y1_l + sigma * Y1_r) / (sigma + sigma_s) * 255
            count += 1

    return Y1


@njit
def get_filtered_file_centered_reverse(samples, sample_filter_left, sample_filter_right, size=500, factor=1):
    """Applies the filters provided into the samples """
    # FIRST FILTER
    l = len(samples)
    samples = np.abs(samples)
    Imax = int(np.amax(samples))

    # full_filter = np.concatenate((sample_filter_left, sample_filter_right))

    Y1 = np.zeros(size)
    count = 0

    for i in np.linspace(0, l, size + 1)[:-1]:
        i = int(i)
        sigma = len(sample_filter_right)
        sigma_s = len(sample_filter_left)
        # print(i)
        d = l - i

        # For the first positions of the filter, cut off filter left
        if i < sigma_s:
            if i == 0:
                Y1[count] = 0
                count += 1

            else:
                sample_filter_left_temp = sample_filter_left[len(sample_filter_left) - i:]
                sigma_s = len(sample_filter_left_temp)

                Y1_l = np.sum(samples[i - len(sample_filter_left_temp):i] * sample_filter_left_temp) / \
                       np.sum(np.full(len(sample_filter_left_temp), Imax * factor) * sample_filter_left_temp)
                Y1_r = Y1_l * (1 - np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right) / \
                               np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right))

                Y1[count] = (sigma_s * Y1_l + sigma * Y1_r) / (sigma + sigma_s) * 255  # OJO! IMPORTANCIA DE CADA LADO?
                count += 1

        # For the last positions of the filter, cut off filter right
        elif d < sigma:
            # if i == l:
            #   Y1[count] = 0
            #  continue
            sample_filter_right_temp = sample_filter_right[:-(len(sample_filter_right) - d)]
            sigma = len(sample_filter_right_temp)

            Y1_l = np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) / \
                   np.sum(np.full(len(sample_filter_left), Imax * factor) * sample_filter_left)
            Y1_r = Y1_l * (1 - np.sum(samples[i:i + len(sample_filter_right_temp)] * sample_filter_right_temp) / \
                           np.sum(np.full(len(sample_filter_right_temp), Imax * factor) * sample_filter_right_temp))
            Y1[count] = (sigma_s * Y1_l + sigma * Y1_r) / (sigma + sigma_s) * 255
            count += 1

        # Middle positions
        else:
            Y1_l = np.sum(samples[i - len(sample_filter_left):i] * sample_filter_left) / \
                   np.sum(np.full(len(sample_filter_left), Imax * factor) * sample_filter_left)
            Y1_r = Y1_l * (1 - np.sum(samples[i:i + len(sample_filter_right)] * sample_filter_right) / \
                           np.sum(np.full(len(sample_filter_right), Imax * factor) * sample_filter_right))
            Y1[count] = (sigma_s * Y1_l + sigma * Y1_r) / (sigma + sigma_s) * 255
            count += 1

    return Y1


def get_matrix_from_file(samples, fs, max_t=5):
    # time = np.arange(0, l) / fs
    max_len = max_t * fs
    # plot_audio(samples, fs)

    l = len(samples)
    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((np.zeros(d), np.asarray(samples)))

    samples = np.abs(samples)

    matrix = []

    for sigma in [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5, 2]:
        for Fs in [1, 1.5, 2, 2.75, 3.5, 4, 4.25, 4.5, 5, 10]:
            sigma_s = sigma / Fs
            h_d = 1
            h_u = h_d / Fs

            sample_filter_left_1 = np.abs(triangle_filter_1_left(sigma_s, fs, h_d))
            sample_filter_right_1 = np.abs(triangle_filter_1_right(sigma, fs, h_u))
            Y1 = get_filtered_file_centered(samples, sample_filter_left_1, sample_filter_right_1)

            sample_filter_left_2 = np.abs(triangle_filter_2_left(sigma, fs, h_u))
            sample_filter_right_2 = np.abs(triangle_filter_2_right(sigma_s, fs, h_d))
            Y2 = get_filtered_file_centered_reverse(samples, sample_filter_left_2, sample_filter_right_2)

            Y1 = np.concatenate((np.full(int(sigma * 100), 0), np.abs(Y1)))
            Y2 = np.concatenate((np.abs(Y2), np.full(int(sigma * 100), 255)))       # 255 so that it does'nt delete the last values
            Y = np.minimum(Y1, Y2)
            # print('shape before minimum, ', Y.shape)

            Y = Y[int(sigma * 100):]
            matrix.append(Y)
            # print('appended y: ', np.asarray(matrix).shape)

    matrix = np.asarray(matrix)
    matrix = np.transpose(matrix)

    print('file matrix shape', matrix.shape)

    return matrix


def get_matrix_from_file_mean(samples, fs, max_t=5):
    """returns a matrix of shape (max_t*100, 100) with the results of applying the filters to an audio file"""
    # time = np.arange(0, l) / fs
    max_len = int(max_t * fs)
    # plot_audio(samples, fs)

    l = len(samples)
    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((np.zeros(d), np.asarray(samples)))

    samples = np.abs(samples)

    matrix = []

    for sigma in [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5, 2]:
        for Fs in [0.5, 1, 1.5, 2, 2.75, 3.5, 4, 4.5, 5, 10]:
            # print('shape before minimum, ', Y.shape)
            Y, _, _ = apply_mean_filter(samples, fs, sigma, Fs)
            matrix.append(Y)
            # print('appended y: ', np.asarray(matrix).shape)

    matrix = np.asarray(matrix)
    matrix = np.transpose(matrix)

    print('file matrix shape', matrix.shape)

    return matrix

# experiments
def experiment_triangle(sigma, Fs, splits_per_second=100, method='mean', shift_factor=0.5, fs=44100,
                        shift=False, save=False):
    import matplotlib.pyplot as plt

    samples = np.concatenate((np.full(fs * 1, 0), triangle_filter_2_left(1, fs, 100), np.zeros(fs)))
    #plot_audio(samples, fs)
    time = np.arange(0, len(samples), 1 / fs)
    sigma_s = sigma / Fs
    h_d = 1
    h_u = h_d / Fs

    if method == 'multiply':
        sample_filter_left_1 = triangle_filter_1_left(sigma_s, fs, h_d)
        sample_filter_right_1 = triangle_filter_1_right(sigma, fs, h_u)
        Y1 = get_filtered_file_multiply(samples, sample_filter_left_1, sample_filter_right_1)
        print(Y1.shape)
        sample_filter_left_2 = triangle_filter_2_left(sigma, fs, h_u)
        sample_filter_right_2 = triangle_filter_2_right(sigma_s, fs, h_d)
        Y2 = get_filtered_file_multiply_reverse(samples, sample_filter_left_2, sample_filter_right_2)
        print(Y2.shape)

        if shift:
            #Y1 = np.concatenate((np.abs(Y1), np.full(int(sigma * 100), 0)))  # 255 so that it does'nt delete the last values
            Y2 = np.concatenate((np.full(int(sigma * shift_factor * 100), 0), Y2[:-int(sigma/2*100)]))
            Y = np.minimum(Y1, Y2)
            # Y = Y[int(sigma * 100):]
        else:
            Y = np.minimum(Y1, Y2)
            # print('shape before minimum, ', Y.shape)

            # Y = Y[int(sigma * 100):]

        fig, axs = plt.subplots(4)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(time, samples)
        axs[1].plot(Y1)
        axs[2].plot(Y2)
        axs[3].plot(Y)

    elif method == 'mean':
        Y, Y_r, Y_l = apply_mean_filter(samples, fs, sigma, Fs, splits_per_second)
        fig, axs = plt.subplots(4)
        fig.suptitle('mean method')
        axs[0].plot(time, samples)
        axs[1].plot(Y_l)
        axs[2].plot(Y_r)
        axs[3].plot(Y)

    else:
        print('wrong specification of method: %s. Select "mean" or "multiply"')

    if save:
        import datetime
        time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")
        name = 'luis'
        # fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))
        fig.savefig('/Users/luisj/Desktop/COUGH/plots/sigma-%s_%s.png' % (sigma, time_stamp))
    else:
        plt.show()

def experiment_one_file(file, sigma, Fs, h_d=1, shift_factor=0.5, save=False, shift=False):
    # plt.plot(np.arange(0,10,1/fs), Y)
    import matplotlib.pyplot as plt

    # file = "/Users/luisj/Desktop/Lcough1.wav"
    audio = AudioSegment.from_wav(file)
    # samples = np.abs(audio.get_array_of_samples())
    if audio.channels == 2:
        audio, _ = audio.split_to_mono()

    samples = audio.get_array_of_samples()
    fs = audio.frame_rate

    l = len(samples)
    max_len = 5 * fs

    print('original length', l / fs)

    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((samples, np.zeros(d)))

    samples = np.abs(samples)
    time = np.arange(0, 5, 1 / fs)
    sigma_s = sigma / Fs
    h_u = h_d / Fs

    sample_filter_left_1 = triangle_filter_1_left(sigma_s, fs, h_d)
    sample_filter_right_1 = triangle_filter_1_right(sigma, fs, h_u)
    Y1 = get_filtered_file_multiply(samples, sample_filter_left_1, sample_filter_right_1)
    print(Y1.shape)
    sample_filter_left_2 = triangle_filter_2_left(sigma, fs, h_u)
    sample_filter_right_2 = triangle_filter_2_right(sigma_s, fs, h_d)
    Y2 = get_filtered_file_multiply_reverse(samples, sample_filter_left_2, sample_filter_right_2)
    print(Y2.shape)

    if shift:
        #Y1 = np.concatenate((np.abs(Y1), np.full(int(sigma * 100), 0)))  # 255 so that it does'nt delete the last values
        Y2 = np.concatenate((np.full(int(sigma * shift_factor * 100), 0), Y2[:-int(sigma/2*100)]))
        Y = np.minimum(Y1, Y2)
        # Y = Y[int(sigma * 100):]
    else:
        Y = np.minimum(Y1, Y2)
    # print('shape before minimum, ', Y.shape)

    # Y = Y[int(sigma * 100):]

    fig, axs = plt.subplots(4)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(time, samples)
    axs[1].plot(Y1)
    axs[2].plot(Y2)
    axs[3].plot(Y)

    if save:
        import datetime
        time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")
        name = 'luis'
        #fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))
        fig.savefig('/Users/luisj/Desktop/COUGH/plots/sigma-%s_%s.png' % (sigma, time_stamp))
    else:
        plt.show()

def experiment_one_file_mean(file, sigma, Fs, save=False, splits_per_second=10):
    # plt.plot(np.arange(0,10,1/fs), Y)
    import matplotlib.pyplot as plt

    # file = "/Users/luisj/Desktop/Lcough1.wav"
    audio = AudioSegment.from_wav(file)
    # samples = np.abs(audio.get_array_of_samples())
    if audio.channels == 2:
        audio, _ = audio.split_to_mono()

    samples = audio.get_array_of_samples()
    fs = audio.frame_rate

    l = len(samples)
    max_len = 5 * fs

    print('original length', l / fs)

    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((samples, np.zeros(d)))

    samples = np.abs(samples)
    time = np.arange(0, 5, 1 / fs)

    Y, _, _ = apply_mean_filter(samples, fs, sigma, Fs, splits_per_second)
    fig, axs = plt.subplots(2)
    fig.suptitle('mean method')
    axs[0].plot(time, samples)
    axs[1].plot(Y)

    if save:
        import datetime
        time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")
        name = 'luis'
        #fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))
        fig.savefig('/Users/luisj/Desktop/COUGH/plots/sigma-%s_%s.png' % (sigma, time_stamp))
    else:
        plt.show()

def test_filter_mean(file, sigma, Fs, sigma_splits, save=False):
    # plt.plot(np.arange(0,10,1/fs), Y)
    import matplotlib.pyplot as plt
    # from pydub import effects

    # file = "/Users/luisj/Desktop/Lcough1.wav"
    audio = AudioSegment.from_wav(file)
    # audio = effects.normalize(audio)
    # normalizedsound.export("./normalized_cough.wav", format="wav")
    # samples = np.abs(audio.get_array_of_samples())
    if audio.channels == 2:
        audio, _ = audio.split_to_mono()

    samples = audio.get_array_of_samples()
    fs = audio.frame_rate
    l = len(samples)
    max_len = 5 * fs

    print('original length', l / fs)

    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((np.zeros(d), samples))

    samples = np.abs(samples)
    time = np.arange(0, 5, 1 / fs)

    Y = apply_mean_filter(samples, fs, sigma, Fs, sigma_splits)

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(time, samples)
    axs[1].plot(Y)

    if save:
        import datetime
        time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")
        name = 'luis'
        fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))
    else:
        plt.show()


def experiment_one_ridge(sigma, Fs, splits_per_second, shift_factor=0.5, method='mean', ridge_length=1, shift=False, save=False):
    # plt.plot(np.arange(0,10,1/fs), Y)
    import matplotlib.pyplot as plt

    fs = 44100
    samples = np.concatenate((np.zeros(fs * 1), np.ones(ridge_length * fs), np.zeros((4 - ridge_length)*fs)))

    l = len(samples)
    max_len = 5 * fs

    print('original length', l / fs)

    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((samples, np.zeros(d)))

    l = len(samples)
    time = np.linspace(0, 5, 5*fs)

    sigma_s = sigma / Fs
    h_d = 1
    h_u = h_d / Fs

    if method == 'multiply':
        sample_filter_left_1 = triangle_filter_1_left(sigma_s, fs, h_d)
        sample_filter_right_1 = triangle_filter_1_right(sigma, fs, h_u)
        Y1 = get_filtered_file_multiply(samples, sample_filter_left_1, sample_filter_right_1)
        print(Y1.shape)
        sample_filter_left_2 = triangle_filter_2_left(sigma, fs, h_u)
        sample_filter_right_2 = triangle_filter_2_right(sigma_s, fs, h_d)
        Y2 = get_filtered_file_multiply_reverse(samples, sample_filter_left_2, sample_filter_right_2)
        print(Y2.shape)

        if shift:
            # Y1 = np.concatenate((np.abs(Y1), np.full(int(sigma * 100), 0)))  # 255 so that it does'nt delete the last values
            Y2 = np.concatenate((np.full(int(sigma * shift_factor * 100), 0), Y2[:-int(sigma / 2 * 100)]))
            Y = np.minimum(Y1, Y2)
            # Y = Y[int(sigma * 100):]
        else:
            Y = np.minimum(Y1, Y2)
            # print('shape before minimum, ', Y.shape)

            # Y = Y[int(sigma * 100):]

        fig, axs = plt.subplots(4)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(time, samples)
        axs[1].plot(Y1)
        axs[2].plot(Y2)
        axs[3].plot(Y)

    elif method == 'mean':
        Y = apply_mean_filter(samples, fs, sigma, Fs, splits_per_second)
        fig, axs = plt.subplots(2)
        fig.suptitle('mean method')
        axs[0].plot(time, samples)
        axs[1].plot(Y)
    else:
        print('wrong specification of method: %s. Select "mean" or "multiply"')

    if save:
        import datetime
        time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")
        name = 'luis'
        # fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))
        fig.savefig('/Users/luisj/Desktop/COUGH/plots/sigma-%s_%s.png' % (sigma, time_stamp))
    else:
        plt.show()


def experiment_triangles(sigma, Fs, splits_per_second, intensity=50, shift_factor=0.5, method='mean', name='luis', shift=False, save=False):
    # plt.plot(np.arange(0,10,1/fs), Y)
    import matplotlib.pyplot as plt

    fs = 44100
    #samples = np.concatenate((np.full(fs, intensity), triangle_filter_2_left(1, fs, 100), np.zeros(fs)))
    samples = np.concatenate((np.full(fs * 1, 0), triangle_filter_2_left(1, fs, 100), np.zeros(fs)))

    l = len(samples)
    max_len = 5 * fs

    print('original length', l / fs)

    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((samples, np.zeros(d)))

    l = len(samples)
    time = np.linspace(0, 5, 5*fs)

    sigma_s = sigma / Fs
    h_d = 1
    h_u = h_d / Fs

    if method == 'multiply':
        sample_filter_left_1 = triangle_filter_1_left(sigma_s, fs, h_d)
        sample_filter_right_1 = triangle_filter_1_right(sigma, fs, h_u)
        Y1 = get_filtered_file_multiply(samples, sample_filter_left_1, sample_filter_right_1)
        print(Y1.shape)
        sample_filter_left_2 = triangle_filter_2_left(sigma, fs, h_u)
        sample_filter_right_2 = triangle_filter_2_right(sigma_s, fs, h_d)
        Y2 = get_filtered_file_multiply_reverse(samples, sample_filter_left_2, sample_filter_right_2)
        print(Y2.shape)

        if shift:
            # Y1 = np.concatenate((np.abs(Y1), np.full(int(sigma * 100), 0)))  # 255 so that it does'nt delete the last values
            Y2 = np.concatenate((np.full(int(sigma * shift_factor * 100), 0), Y2[:-int(sigma / 2 * 100)]))
            Y = np.minimum(Y1, Y2)
            # Y = Y[int(sigma * 100):]
        else:
            Y = np.minimum(Y1, Y2)
            # print('shape before minimum, ', Y.shape)

            # Y = Y[int(sigma * 100):]

        fig, axs = plt.subplots(4)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(time, samples)
        axs[1].plot(Y1)
        axs[2].plot(Y2)
        axs[3].plot(Y)

    elif method == 'mean':
        Y, Y_r, Y_l = apply_mean_filter(samples, fs, sigma, Fs, splits_per_second)
        fig, axs = plt.subplots(4)
        fig.suptitle('mean method')
        axs[0].plot(time, samples)
        axs[1].plot(Y_r)
        axs[2].plot(Y_l)
        axs[3].plot(Y)
    else:
        print('wrong specification of method: %s. Select "mean" or "multiply"')

    if save:
        import datetime
        time_stamp = datetime.datetime.now().strftime("%I:%M:%S-%p")
        # fig.savefig('/store/experiments/covid/plots/%s_%s.png' % (name, time_stamp))
        fig.savefig('/Users/luisj/Desktop/COUGH/plots/new/sigma-%s_%s_%s.png' % (str(sigma), time_stamp, name))
    else:
        plt.show()



def debug_function():
    # file = "/store/datasets/covid/examples/cough5s.wav"
    file = "/Users/luisj/Desktop/Lcough1.wav"
    # experiment_one_file(file, 0.3, 4)
    audio = AudioSegment.from_wav(file)
    if audio.channels == 2:
        audio, _ = audio.split_to_mono()
    samples = audio.get_array_of_samples()
    fs = audio.frame_rate

    matrix = get_matrix_from_file(samples, fs)
    print(matrix)


def experiment_one_matrix(file="/store/datasets/covid/examples/cough5s.wav"):
    import matplotlib.pyplot as plt

    # file = "/Users/luisj/Desktop/Lcough1.wav"
    audio = AudioSegment.from_wav(file)
    # samples = np.abs(audio.get_array_of_samples())
    if audio.channels == 2:
        audio, _ = audio.split_to_mono()

    samples = audio.get_array_of_samples()
    fs = audio.frame_rate

    max_len = int(5 * fs)
    # plot_audio(samples, fs)

    l = len(samples)
    if max_len < l:
        samples = samples[:max_len]
        samples = np.asarray(samples)
    elif l < max_len:
        d = max_len - l
        samples = np.concatenate((np.zeros(d), np.asarray(samples)))

    matrix = get_matrix_from_file_mean(samples, fs)

    # print(np.max(matrix))
    #plot_mfcc(matrix)

    time = np.linspace(0, 5, max_len)
    fig, axs = plt.subplots(2)
    fig.suptitle('mean method')
    axs[0].plot(time, samples)
    axs[1].imshow(np.transpose(matrix), interpolation="nearest", origin="upper")

    plt.show()


def main_triangles_1(language):
    """Applies filters to all files in 'language' directory and saves resulting spectrum to job"""
    # Destination
    dst = '/store/datasets/jobs/'

    if 'filtered_languages' not in os.listdir(dst):
        os.mkdir(dst + 'filtered_languages')
    dst += 'filtered_languages/'

    if language not in os.listdir(dst):
        os.mkdir(dst + language)
    dst += language + '/'

    # Here we will store file names
    fails = []
    files_p = []
    files_n = []

    # POSITIVES
    """folder = '/store/datasets/covid/audiosl/languages/%s/positives/' % language
    files = os.listdir(folder)
    # file = "/store/datasets/covid/examples/cough5s.wav"

        # When working with sentiment dataset
    if language.find('entiment') != -1:
        folder = '/store/datasets/aux/Audio_Speech_Actors_01-24/'
        names, _ = get_sentiment_files()
        files = [str(f) for f in names]

    X_pos = []
    count = 0
    for f in files:
        try:
            # file = '/Users/luisj/Downloads/cough5s.wav'
            audio = AudioSegment.from_wav(folder + f)
            if audio.channels == 2:
                audio, _ = audio.split_to_mono()
            samples = audio.get_array_of_samples()
            fs = audio.frame_rate

            X_pos.append(get_matrix_from_file(samples, fs))
            files_p.append(f)
            joblib.dump(X_pos, dst + "X_%s.job" % language)
            count += 1
            print('Positive files: ', count, np.asarray(X_pos).shape)

        except Exception:
            print('skipped file: ', f)
            fails += f

        with open(dst + '%s_fail_files.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(fails)

        with open(dst + '%s_positives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_p)

        with open(dst + '%s_negatives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_n)

        # if count % 100 == 0:
        #    print("positive data hundreds = ", count / 100)

    if language.find('entiment') != -1:
        exit()"""

    # NEGATIVES
    folder = '/store/datasets/covid/audiosl/languages/%s/negatives/' % language
    # folder = '/store/datasets/aux/Audio_Speech_Actors_01-24/Actor_04/'
    files = os.listdir(folder)

    # file = "/store/datasets/covid/examples/cough5s.wav"

    X_neg = []
    count = 0
    for f in files:
        try:
            audio = AudioSegment.from_wav(folder + f)
            if audio.channels == 2:
                audio, _ = audio.split_to_mono()
            samples = audio.get_array_of_samples()
            fs = audio.frame_rate

            X_neg.append(get_matrix_from_file(samples, fs))
            files_n.append(f)
            count += 1
            joblib.dump(X_neg, dst + "Xn_%s.job" % language)
            print('negative files:', count, np.asarray(X_neg).shape)
        except Exception:
            print('skipped file: ', f)
            fails += f

        with open(dst + '%s_fail_files.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(fails)

        with open(dst + '%s_positives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_p)

        with open(dst + '%s_negatives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_n)

    return None

def apply_filters_casa(language, dst = '/Volumes/TOSHIBA EXT/Projetto Conclusso/Azure_back-up/Datasets/mean_filters_0/'):
    """Applies normalized filters to all files in 'language' directory and saves resulting spectrum to job"""
    # Load parameters
    import json
    with open('parameters.json') as f:
        variables = json.load(f)

    if variables['save_name'] not in os.listdir(dst):
        os.mkdir(dst + variables['save_name'])
    dst += variables['save_name'] + '/'

    if language not in os.listdir(dst):
        os.mkdir(dst + language)
    dst += language + '/'

    # Here we will store file names
    fails = []
    files_p = []
    files_n = []

    # POSITIVES

    # file = "/store/datasets/covid/examples/cough5s.wav"
    # When working with sentiment dataset
    if language.find('entiment') != -1:
        folder = '/store/datasets/aux/Audio_Speech_Actors_01-24/'
        names, _ = get_sentiment_files()
        files = [str(f) for f in names]
    elif language.find('peech') != -1:
        folder = '/store/datasets/aux/all_librispeech_links/'
        files = os.listdir(folder)
    else:
        folder = DB_DIR + '%s/positives/' % language
        files = os.listdir(folder)

    X_pos = []
    count = 0
    for f in files:
        try:
            # file = '/Users/luisj/Downloads/cough5s.wav'
            audio = AudioSegment.from_wav(folder + f)
            if audio.channels == 2:
                audio, _ = audio.split_to_mono()
            samples = audio.get_array_of_samples()
            fs = audio.frame_rate

            X_pos.append(get_matrix_from_file_mean(samples, fs))
            files_p.append(f)
            joblib.dump(X_pos, dst + "Xp_%s.job" % language)
            count += 1
            print('Positive files: ', count, np.asarray(X_pos).shape)

        except Exception:
            print('skipped file: ', f)
            fails += f

        if fails != []:
            with open(dst + '%s_fail_files.csv' % language, 'w') as file:
                write = csv.writer(file)
                write.writerow(fails)

        with open(dst + '%s_positives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_p)

        with open(dst + '%s_negatives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_n)

        # if count % 100 == 0:
        #    print("positive data hundreds = ", count / 100)

    if language.find('entiment') != -1 or language.find('peech') != -1:
        exit()
    elif language.find('peech') != -1:
        exit()

    # NEGATIVES
    folder = DB_DIR + '%s/negatives/' % language
    # folder = '/store/datasets/aux/Audio_Speech_Actors_01-24/Actor_04/'
    files = os.listdir(folder)

    # file = "/store/datasets/covid/examples/cough5s.wav"

    X_neg = []
    count = 0
    for f in files:
        try:
            audio = AudioSegment.from_wav(folder + f)
            if audio.channels == 2:
                audio, _ = audio.split_to_mono()
            samples = audio.get_array_of_samples()
            fs = audio.frame_rate

            X_neg.append(get_matrix_from_file_mean(samples, fs))
            files_n.append(f)
            count += 1
            joblib.dump(X_neg, dst + "Xn_%s.job" % language)
            print('negative files:', count, np.asarray(X_neg).shape)

        except Exception:
            print('skipped file: ', f)
            fails += f

        if fails != []:
            with open(dst + '%s_fail_files.csv' % language, 'w') as file:
                write = csv.writer(file)
                write.writerow(fails)

        with open(dst + '%s_positives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_p)

        with open(dst + '%s_negatives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_n)

    return None


def main(language, dst = '/store/datasets/jobs/'):
    """Applies normalized filters to all files in 'language' directory and saves resulting spectrum to job"""
    # Load parameters
    import json
    with open('parameters.json') as f:
        variables = json.load(f)

    if variables['save_name'] not in os.listdir(dst):
        os.mkdir(dst + variables['save_name'])
    dst += variables['save_name'] + '/'

    if language not in os.listdir(dst):
        os.mkdir(dst + language)
    dst += language + '/'

    # Here we will store file names
    fails = []
    files_p = []
    files_n = []

    # POSITIVES

    # file = "/store/datasets/covid/examples/cough5s.wav"
    # When working with sentiment dataset
    if language.find('entiment') != -1:
        folder = '/store/datasets/aux/Audio_Speech_Actors_01-24/'
        names, _ = get_sentiment_files()
        files = [str(f) for f in names]
    elif language.find('peech') != -1:
        folder = '/store/datasets/aux/all_librispeech_links/'
        files = os.listdir(folder)
    else:
        folder = '/store/datasets/covid/audiosl/languages/%s/positives/' % language
        files = os.listdir(folder)

    X_pos = []
    count = 0
    for f in files:
        try:
            # file = '/Users/luisj/Downloads/cough5s.wav'
            audio = AudioSegment.from_wav(folder + f)
            if audio.channels == 2:
                audio, _ = audio.split_to_mono()
            samples = audio.get_array_of_samples()
            fs = audio.frame_rate

            X_pos.append(get_matrix_from_file_mean(samples, fs))
            files_p.append(f)
            joblib.dump(X_pos, dst + "Xp_%s.job" % language)
            count += 1
            print('Positive files: ', count, np.asarray(X_pos).shape)

        except Exception:
            print('skipped file: ', f)
            fails += f

        if fails != []:
            with open(dst + '%s_fail_files.csv' % language, 'w') as file:
                write = csv.writer(file)
                write.writerow(fails)

        with open(dst + '%s_positives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_p)

        with open(dst + '%s_negatives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_n)

        # if count % 100 == 0:
        #    print("positive data hundreds = ", count / 100)

    if language.find('entiment') != -1 or language.find('peech') != -1:
        exit()
    elif language.find('peech') != -1:
        exit()

    # NEGATIVES
    folder = '/store/datasets/covid/audiosl/languages/%s/negatives/' % language
    # folder = '/store/datasets/aux/Audio_Speech_Actors_01-24/Actor_04/'
    files = os.listdir(folder)

    # file = "/store/datasets/covid/examples/cough5s.wav"

    X_neg = []
    count = 0
    for f in files:
        try:
            audio = AudioSegment.from_wav(folder + f)
            if audio.channels == 2:
                audio, _ = audio.split_to_mono()
            samples = audio.get_array_of_samples()
            fs = audio.frame_rate

            X_neg.append(get_matrix_from_file_mean(samples, fs))
            files_n.append(f)
            count += 1
            joblib.dump(X_neg, dst + "Xn_%s.job" % language)
            print('negative files:', count, np.asarray(X_neg).shape)

        except Exception:
            print('skipped file: ', f)
            fails += f

        if fails != []:
            with open(dst + '%s_fail_files.csv' % language, 'w') as file:
                write = csv.writer(file)
                write.writerow(fails)

        with open(dst + '%s_positives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_p)

        with open(dst + '%s_negatives.csv' % language, 'w') as file:
            write = csv.writer(file)
            write.writerow(files_n)

    return None

def trash():
    # file = "/store/datasets/covid/examples/cough5s.wav"
    #from streamlit import caching
    #caching.clear_cache()
    #file_1 = '/Users/luisj/Downloads/Lcough1.wav'
    #file_2 = '/Users/luisj/Downloads/2514-149482-0009.wav'
    # samples = np.concatenate((np.full(44100 * 1, 0), triangle_filter_2_left(0.5, 44100, 100), np.zeros(44100)))
    # plot_audio(samples, 44100)
    #for sigma in [0.2, 0.5, 0.7, 1, 1.5, 2, 2.2, 2.5]:
    #experiment_triangles(1, 10, 100, intensity=sigma, shift=True)
    #experiment_one_file_mean(file_1, 0.5, 5, save=False, splits_per_second=10)
    # experiment_one_ridge(1, 4, 100, ridge_length=2, shift=True)
    # experiment_one_file_mean(file_1, 2, 5, save=False, splits_per_second=2)
    # experiment_one_matrix(file_1)
    #for sigma in [0.2, 0.4, 0.5, 0.7, 1, 1.5, 2]:
     #   experiment_triangles(sigma, sigma*10, 100, shift=False, method='mean', save=True)

    # debug_function()

    # audio = AudioSegment.from_wav(file)
    # samples = np.abs(audio.get_array_of_samples())
    # if audio.channels == 2:
    #    audio, _ = audio.split_to_mono()

    # samples = audio.get_array_of_samples()
    # fs = audio.frame_rate

    # plot_audio(samples, fs)
    # SEE ONE MATRIX
    # experiment_one_ridge(1, 4)
    # test_filter_mean(file, 0.5, 2, 100)
    # fs = 44100
    # samples = np.zeros(5*fs)

    #matrix = get_matrix_from_file_mean(samples, fs)
    #plot_mfcc(matrix)
    #exit()

    # np.save("filter_Lcough1", matrix)
    # matrix = np.ones((100,100))
    # matrix = np.load('filter_Lcough1.npy')
    # joblib.dump(matrix, "onefiledumped.job")

    # matrix = joblib.load('/Users/luisj/Downloads/onefiledumped.job')
    # matrix = np.transpose(matrix)

    """file = '/Users/luisj/Downloads/cough5s.wav'
    audio = AudioSegment.from_wav(file)
    if audio.channels == 2:
        audio, _ = audio.split_to_mono()
    samples = audio.get_array_of_samples()
    fs = audio.frame_rate

    X = apply_mean_filter(samples, fs, 0.5, 0.5)
    print(X.shape)
    print(X)"""
    # exit()
    # files, _ = get_sentiment_files()
    # print(files)
    return None

if __name__ == '__main__':

    language = 'main'
    print('working with dataset: ', language)
    apply_filters_casa(language)
