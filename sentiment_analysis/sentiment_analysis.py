""" NOte to reader: This was one of my first ML project, so please do not judge. Many things could be improved here"""

import os
import sys
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications
import pandas as pd


# Change this to the location of the database directories
DB_DIR = os.path.dirname(os.path.realpath(__file__))

# Import databases
sys.path.insert(1, DB_DIR)
from db_utils import get_sentiment_data #, PlotMfcc

def Secure_Voice_Channel(func):
    """Define Secure_Voice_Channel decorator."""
    def execute_func(*args, **kwargs):
        print('Established Secure Connection.')
        returned_value = func(*args, **kwargs)
        print("Ended Secure Connection.")

        return returned_value

    return execute_func

@Secure_Voice_Channel
def generic_vns_function(input_shape, units, lr, classes=8):
    """Generic Deep Learning Model generator."""

    base_model = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape, pooling=None,
                                         classes=classes)

    x = base_model.output

    x = layers.GlobalAveragePooling2D()(x)     

    # x = layers.Flatten()(x)                     #Alternative to GlobalAveragePooling2D


    predictions = layers.Dense(units, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    opt = Adam(lr=lr)

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size, X_train, y_train, X_val, Y_val, X_test, y_test):
    """Generic Deep Learning Model training function."""

    model.fit(X_train, y_train, validation_data=(X_val,Y_val), epochs=epochs,
              batch_size=batch_size, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=2)

    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    return model

def import_sentiment_dataset():
    X, Y = get_sentiment_data()
    m = len(X)

    
    # Note: can be done quicker with sklearn split_dataset function, and randomization would not hur either. But this is an old project
    X_train, Y_train = X[0:int(m*0.7)], Y[0:int(m*0.7)]  # 70% training
    X_val, Y_val = X[int(m*0.7):int(m*0.85)], Y[int(m*0.7):int(m*0.85)]  #15% Validation
    X_test, Y_test = X[int(m*0.85):m], Y[int(m*0.85):m]       # 15% testing

    X_train, X_test, X_val = normalize_dataset(X_train, X_test, X_val)
    (X_train, Y_train), (X_test, Y_test), (X_val, Y_val) = reshape_dataset(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def normalize_dataset(X_train, X_test, X_val):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train-mean)/std
    X_test = (X_test-mean)/std
    X_val = (X_val-mean)/std

    return X_train, X_test, X_val

def reshape_dataset(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    """Reshape dataset for Convolution."""
    num_pixels = X_test.shape[1]*X_test.shape[2]

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1).astype('float32')

    #Ys are 0 or 1, not categorical

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)


def main():

    # Hyperparameters

    layer_units = 1000
    epochs = 3
    batch_size = 240
    lr = 0.0001

    # Import Datasets
    (X_train, Y_train, X_val, Y_val, X_test, Y_test) = import_sentiment_dataset()


    # Generate and train model
    model = generic_vns_function(X_train.shape[1:], Y_train.shape[1], layer_units, lr)
    #print(model.summary())

    model_name = "SA_1Out"

    model_json = model.to_json()
    with open("%s.json" %model_name, "w") as json_file:
        json_file.write(model_json)

    trained_model = train_model(model, epochs, batch_size, X_train, Y_train, X_val, Y_val, X_test, Y_test)


    # Save model to h5 file
    trained_model.save('models/model_%s_a1.h5' %model_name)
    model.save_weights('%s_weights.h5' %model_name)

    print("Saved model to disk")

    return None

if __name__ == '__main__':
    main()
