import numpy
import os
import scipy.io.wavfile
import python_speech_features
import re
import wave
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

from pydub import AudioSegment

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def PlotMfcc(mfcc_data):        #Not used
    from matplotlib import cm
    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(mfcc_data, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')

    plt.show()

    return None

def save_dataframe(X_train, Y_train, X_test, Y_test):           #Not working
    """No funciona so far por una cuestión de dimensions"""

    Xtr = pd.Series(X_train)
    Ytr = pd.Series(Y_train)

    Xte = pd.Series(X_test)
    Yte = pd.Series(Y_test)

    df = pd.DataFrame({ 'X train': Xtr, 'Y train': Ytr, 'X test': Xte, 'Y test': Yte})
    df.to_csv(index=False)

    return None

def PlotAudio(data, fs):

    time = np.arange(0,len(data))/fs

    plt.plot(time,data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time-domain signal')
    plt.show()

    return None

def match_wavs(f1, f2):
    sound_file_1 = AudioSegment.from_wav(f1)
    samples_1 = sound_file_1.get_array_of_samples()

    sound_file_2 = AudioSegment.from_wav(f2)
    samples_2 = sound_file_2.get_array_of_samples()

    n1 = len(samples_1)
    n2 = len(samples_2)

    if n1 > n2:
        z = np.zeros(n1-n2)
        samples_2.append(z)
        print(len(samples_2), len(samples_1))

    elif n2 > n1:
        z = np.zeros(n1 - n2)
        samples_1.append(z)
        print(len(samples_2), len(samples_1))

    else:
        print('samples have equal lengths')

    return samples_1, samples_2


def from_mp3_dir_to_wav():
    from glob import glob
    import librosa
    import soundfile as sf


    # files
    src = "/Users/luisj/Desktop/SentimentAnalysis/dataset-main/raw"
    dst = "/Users/luisj/Desktop/SentimentAnalysis/dataset-main/WAV"

    af = glob(src +'/*.mp3')
    #audios = np.asarray(af)
    #print(audios.shape)



    for f in range(len(af)):
        a, fs = librosa.load(af[f])
        sf.write(dst+'/%s.wav' %f, a, fs)
        print(f)
    #PlotAudio(a, fs)

    return None


def from_mp3_file_to_wav(file,src, dst):
    import librosa
    import soundfile as sf

    # files
    #src = "/Users/luisj/Desktop/SentimentAnalysis/dataset-main/raw"
    #dst = "/Users/luisj/Desktop/SentimentAnalysis/dataset-main/WAV"


    a, fs = librosa.load(src+'/%s' %file)
    new_filename = file.replace('.mp3','.wav')
    sf.write(dst + '/' + new_filename, a, fs)

    # PlotAudio(a, fs)

    return None


################################################################################
######Sentiment Analysis dataset  #####################
################################################################################
class DataFetcherSentimentSpeech:

    def __init__(self, path, cepstrum_dimension,
     winlen=0.020, winstep=0.01, encoder_filter_frames= 8, encoder_stride= 5):
        self.path = path
        self.cepstrum_dimension = cepstrum_dimension
        self.winlen = winlen
        self.winstep = winstep
        self.recording = False
        self.frames = numpy.array([], dtype=numpy.int16)
        self.samplerate = 16000
        self.encoder_stride = encoder_stride
        self.encoder_filter_frames = encoder_filter_frames

    def get_mcc_from_file(self, f, max_size):

        mfcc_array = []
        counter = 0


        (rate,sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0])/(rate*max_size/100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0]//ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n+inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension, nfft=2048, lowfreq=0, highfreq=16000/2)
                if mfcc.shape[0]< max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0),(0,0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension, nfft=2048, lowfreq=0, highfreq=16000/2)
        if mfcc.shape[0]< max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0),(0,0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array

    def get_mcc_from_audio(self, sig, rate, max_size):

        mfcc_array = []
        counter = 0


        # (rate,sig) = scipy.io.wavfile.read(f)
        ratio = (sig.shape[0])/(rate*max_size/100)
        if ratio > 1:
            sigs = []
            n = 0
            inc = int(sig.shape[0]//ratio)
            for i in range(int(np.trunc(ratio))):
                new_sig = sig[n:(n+inc)]
                n += inc
                mfcc = python_speech_features.mfcc(new_sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension, nfft=1024, lowfreq=0, highfreq=16000/2)
                if mfcc.shape[0]< max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0),(0,0)), mode='constant')
                elif mfcc.shape[0] > max_size:
                    mfcc = mfcc[0:max_size]

                mfcc_array.append(mfcc)
            return mfcc_array

        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension, nfft=1024, lowfreq=0, highfreq=16000/2)
        if mfcc.shape[0]< max_size:
            mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0),(0,0)), mode='constant')
        mfcc_array.append(mfcc)

        return mfcc_array

#OUTDATED, NOT USED
def get_covid_files(path = DIR_PATH):   #Outdated
    '''Returns the pandas series with files names and the series y with labels, where [1,0] = positive '''
    X=[]
    Y=[]
    json_path = path + '/dataset-main/metadata.json'
    df = pd.read_json(json_path)
    df = df.sample(frac=1).reset_index(drop=True)   #Arrange rows in random order

    for file in df.filename:
        X.append(file)

    X = np.asarray(X)

    for c in df.covid19:
        if c == True:
            Y.append([1, 0])
        elif c==False:
            Y.append([0, 1])

    print('m =', len(X))

    return X, Y




def get_sentiment_files(source_path=DIR_PATH):
    '''Returns the files within the Audio_Speech directories'''
    authors = [f for f in os.listdir(source_path + '/Audio_Speech_Actors_01-24/')]
    print(authors)
    books = []
    Y = []
    for author in authors:
        books.extend([author + '/' + f for f in os.listdir(source_path + '/Audio_Speech_Actors_01-24/' + author + '/')])


    for book in books:
        idx = book.find('-')
        book = book[idx+1:]

        idx = book.find('-')
        #print(book[idx+1:idx+3])
        Y.append(book[idx+1:idx+3])

    return books, Y

def get_sentiment_data():
    '''Returns the mfcc of the files (x) and its correspondig sentiment label (y)'''
    files, Y = get_sentiment_files()
    audio_data = DataFetcherSentimentSpeech(DIR_PATH, 50)


    X = []
    for file in files:
        file = DIR_PATH + '/Audio_Speech_Actors_01-24/' + file

        sound_file = AudioSegment.from_wav(file)
        audio = sound_file.get_array_of_samples()
        # audio[0] = sound_file.frame_rate
        audios = audio_data.get_mcc_from_audio(np.asarray(audio), sound_file.frame_rate, 500)
        # audios = audio_data.get_mcc_from_auido(np.asarray(sound_file), sound_file.samplerate 100)
        X.append(audios[0])

    y_one_hot = []
    '''for y in Y:
        if y == '01':
            y_one_hot.append([1,0,0,0,0,0,0,0])
        elif y == '02':
            y_one_hot.append([0,1,0,0,0,0,0,0])
        elif y == '03':
            y_one_hot.append([0,0,1,0,0,0,0,0])
        elif y == '04':
            y_one_hot.append([0,0,0,1,0,0,0,0])
        elif y == '05':
            y_one_hot.append([0,0,0,0,1,0,0,0])
        elif y == '06':
            y_one_hot.append([0,0,0,0,0,1,0,0])
        elif y == '07':
            y_one_hot.append([0,0,0,0,0,0,1,0])
        elif y == '08':
            y_one_hot.append([0,0,0,0,0,0,0,1]) #Originalmente [1,0,0,0,0,0,0,1]
        else:
            print(y)'''

    '''for y in Y:
            if y == '01':
                y_one_hot.append([1,0])
            elif y == '02':
                y_one_hot.append([1,0])
            elif y == '03':
                y_one_hot.append([1,0])
            elif y == '04':
                y_one_hot.append([1,0])
            elif y == '05':
                y_one_hot.append([0,1])
            elif y == '06':
                y_one_hot.append([1,0])
            elif y == '07':
                y_one_hot.append([1,0])
            elif y == '08':
                y_one_hot.append([1,0]) #Originalmente [1,0,0,0,0,0,0,1]
            else:
                print(y)'''

    for y in Y:
        if y == '01':
            y_one_hot.append([0])
        elif y == '02':
            y_one_hot.append([0])
        elif y == '03':
            y_one_hot.append([0])
        elif y == '04':
            y_one_hot.append([0])
        elif y == '05':
            y_one_hot.append([1])
        elif y == '06':
            y_one_hot.append([0])
        elif y == '07':
            y_one_hot.append([0])
        elif y == '08':
            y_one_hot.append([0])  # Originalmente [1,0,0,0,0,0,0,1]
        else:
            print(y)

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

def main():
    '''for testing'''
    X, Y = get_sentiment_files()
    print(X)
    print(Y)

    return None

if __name__ == '__main__':
    main()
