### IMPORT NECESSARY LIBRARIES ###
# basic libraries:
import os
import numpy as np
import time
import joblib
import pandas as pd

# audio libraries:
import python_speech_features
import scipy.io.wavfile
from pydub import AudioSegment
import librosa

# PATH OF OUR CURRENT PROJECT
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# PATH OF THE DATABASE DIRECTORIES
DB_DIR = '/store/datasets/covid/audiosl'


### FUNCTIONS ###
def plot_audio(samples, fs):
    """Plot an audio signal over time (in seconds)"""
    import matplotlib.pyplot as plt
    time = np.arange(0, len(samples)) / fs

    plt.plot(time, samples)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time-domain signal')
    plt.show()

    return None


# MFCC

# simple
def simple_mfcc(samples, fs):
    samples = np.array(samples, dtype=float)
    mfccs = librosa.feature.mfcc(samples, sr=fs)
    print("shape of the mfcc is: ", mfccs.shape)

    return mfccs


# complex:
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


def mfcc_maker(files, time, cepstrum_dimension):
    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension)  # previously DB_DIR

    X = []
    fails = []
    for file in files:
        try:
            sound_file = AudioSegment.from_wav(file)
            audio = sound_file.get_array_of_samples()
            fs = sound_file.frame_rate

            audios = audio_data.get_mcc_from_audio(np.asarray(audio), fs, time * 100)
            # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
            X.append(audios[0])

        except Exception:
            print("Failed to load:", file)
            fails.append(file)

    return X, fails


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


### RECORD AUDIO ###

# define the countdown/count-up func.
def countdown(t):
    while t!=0:
        print(t)
        time.sleep(1)
        t -= 1
def countup(t):
    c=0
    while c!=t+1:
        print(c)
        time.sleep(1)
        c += 1


def audio_recording(fs=44100, seconds=3, filename='output.wav'):
    import sounddevice as sd
    from scipy.io.wavfile import write

    print("Will record for %i seconds: " % seconds)
    input("press any key to start recording for %i seconds" % seconds)

    print("begin in:")
    countdown(int(3))
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.dtype('int16'))
    countup(int(seconds))
    sd.wait()  # Wait until recording is finished
    print('recording finished')

    write(filename, fs, myrecording)  # Save as WAV file
    return myrecording


############################################
############################################

def test_mfcc():
    file = "/Users/luisj/Downloads/THEM3.wav"
    sound_file = AudioSegment.from_wav(file)

    audio = sound_file.get_array_of_samples()
    fs = sound_file.frame_rate
    # plot_audio(audio, fs)
    mfcc = simple_mfcc(audio, fs)
    plot_mfcc(mfcc)

    audio_data = DataFetcher(DIR_PATH, cepstrum_dimension=63)
    mfcc = audio_data.get_mcc_from_audio(audio, fs, 20)
    plot_mfcc(mfcc)
    return None


def main():
    rec = audio_recording()
    plot_audio(rec, 44100)

if __name__ == '__main__':
    main()
