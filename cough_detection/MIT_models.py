import os
import sys
import numpy as np
# import datetime

from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, SGD
# from tensorflow.keras.datasets import mnist
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger

from sklearn.model_selection import train_test_split
import tensorflow as tf

# import h5py
# import tensorflow.keras.backend as K
# from tensorflow.python.keras.saving.saved_model import load as saved_model_load
# from tensorflow.python.saved_model import loader_impl
# from tensorflow.keras.metrics import FalseNegatives, FalsePositives

# Change this to the location of the database directories
# DB_DIR = os.path.dirname(os.path.realpath(__file__)) # + "/../databases"
DB_DIR = '/store/datasets/covid/audiorec'

# Import databases
sys.path.insert(1, DB_DIR)

from db_utils import get_sentiment_data, \
    get_librispeech_wakeword  # , get_dementia_gender, get_librispeech_gender, get_cough_ENG_CAT_from_pickle, get_cough_from_pickle, get_speech_dataset_complete, get_dementia_dataset, get_librispeech_dataset_word_count, get_librispeech_dataset_word_segmented


def load_JSON_and_weights(json_path, weights_path):
    """Loads Json architecture and weight to return a trained model"""

    # Load Json Architecture
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = models.model_from_json(loaded_model_json)

    # Load weights
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")

    return loaded_model


def normalize_dataset(X_train, X_test):
    """Normalize speech recognition and computer vision datasets."""
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - std) / mean
    X_test = (X_test - std) / mean
    return X_train, X_test


def poissonw_noise(img, weight=1):
    noise_mask = (np.random.poisson(np.abs(img * 255.0 * weight)) / weight / 255).astype(float)
    return noise_mask


def reshape_dataset(X_train, y_train, X_test, y_test):
    """Reshape dataset for Convolution."""

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)


def train_model_full(model, X_train, y_train, save_dir, language, val_data, monitor, epochs=100,
                     batch_size=100, optim='adam', lr=0.001, patience=5):
    """Train provided model with speech training dataset"""

    """Define Metrics"""
    falsen = tf.keras.metrics.FalseNegatives()
    falsep = tf.keras.metrics.FalsePositives()

    """Select Optimizer"""
    if optim is 'adam':
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy', falsep, falsen])

    elif optim is 'nadam':
        optimizer = Nadam(learning_rate=lr)
        model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy', falsep, falsen])

    elif optim is 'SGD':
        optimizer = SGD(learning_rate=lr)
        model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy', falsep, falsen])

    elif optim is 'adamax':
        optimizer = Adamax(learning_rate=lr)
        model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy', falsep, falsen])

    elif optim is 'None':
        print("using previously compiled model")

    else:
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy', falsep, falsen])

    """Define Model Name to be Saved"""
    model_save = save_dir + language + '.h5'

    """Set a checkpoint to save the model whenever the monitored value improves"""
    checkpoint = callbacks.ModelCheckpoint(model_save, monitor=monitor, verbose=1, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

    """Set conditions for early stopping the training"""
    csv_logger = CSVLogger(save_dir + "model_history.csv", append=True)
    early = callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=1, mode='auto')
    cb = [checkpoint, early, csv_logger]

    """Fit model"""
    history = model.fit(X_train, y_train, validation_data=val_data, epochs=epochs,
                        batch_size=batch_size, verbose=1, callbacks=cb)

    """Return model path and training history"""
    return history, model_save


def train_model_full_steps(model, X_train, y_train, save_dir, language, val_data, monitor, epochs=100,
                           steps=150, optim='adam', lr=0.001, patience=5):
    """Train provided model with a specific number of steps per epoch"""

    """Select Optimizer"""
    if optim is 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optim is 'nadam':
        optimizer = Nadam(learning_rate=lr)
    elif optim is 'SGD':
        optimizer = SGD(learning_rate=lr)
    elif optim is 'adamax':
        optimizer = Adamax(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)

    """Define Metrics"""
    falsen = tf.keras.metrics.FalseNegatives()
    falsep = tf.keras.metrics.FalsePositives()

    """Compile Model with chosen optimizer, loss and metrics"""
    model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy', falsep, falsen])

    """Define Model Name to be Saved"""
    model_save = save_dir + language + '.h5'

    # model_save = "models/{}.h5".format(model_file_name)

    """Set a checkpoint to save the model whenever the monitored value improves"""
    checkpoint = callbacks.ModelCheckpoint(model_save, monitor=monitor, verbose=1, save_best_only=True,
                                           save_weights_only=False, mode='auto', save_freq='epoch')

    """Set conditions for early stopping the training"""
    csv_logger = CSVLogger(save_dir + "model_history.csv", append=True)
    early = callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=1, mode='auto')
    cb = [checkpoint, early, csv_logger]

    #cb = [checkpoint, early, csv_logger]

    """Fit model"""
    history = model.fit(X_train, y_train, validation_data=val_data, epochs=epochs,
                        steps_per_epoch=steps, verbose=1, callbacks=cb)
    """Return model and training history"""
    return model, model_save


def train_model_WO(model, X_train, y_train, save_dir, language, val_data, epochs=100,
                   batch_size=100, optim='adam', lr=0.001, patience=10, sub_dir="models"):
    """Train provided model with training dataset. Save weights only"""

    if sub_dir not in os.listdir(save_dir):
        os.mkdir(save_dir + sub_dir)

    """Select Optimizer"""
    if optim is 'adam':
        optimizer = Adam(learning_rate=lr)
    elif optim is 'nadam':
        optimizer = Nadam(learning_rate=lr)
    else:
        optimizer = Adamax(learning_rate=lr)

    """Define Metrics"""
    falsen = tf.keras.metrics.FalseNegatives()
    falsep = tf.keras.metrics.FalsePositives()

    """Compile Model with chosen optimizer, loss and metrics"""
    model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy', falsep, falsen])

    """Define Model Name to be Saved"""
    save_name = language + '_weights.h5'
    model_save = save_dir + 'models/' + save_name

    # model_save = "models/{}.h5".format(model_file_name)

    """Set a checkpoint to save the model whenever the validation accuracy improves"""
    checkpoint = callbacks.ModelCheckpoint(model_save, monitor='val_accuracy', verbose=1, save_best_only=True,
                                           save_weights_only=False, mode='auto', period=1)

    """Set conditions for early stopping the training"""
    early = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=patience, verbose=1, mode='auto')
    cb = [checkpoint, early]

    """Fit model"""
    history = model.fit(X_train, y_train, validation_data=val_data, epochs=epochs,
                        batch_size=batch_size, verbose=1, callbacks=cb)

    """Return model and training history"""
    return history, save_name


def choose_model(input_shape, classes=1, model_name="ResNet50", fine_tune=False, N_layers=0) -> object:
    """Choose model from model_name string."""

    if model_name == "VGG16":
        base_model = applications.VGG16(include_top=False, weights=None, input_shape=input_shape, pooling=None)
    elif model_name == "DenseNet201":
        base_model = applications.DenseNet201(include_top=False, weights=None, input_shape=input_shape, pooling=None)
    elif model_name == "ResNet50":
        base_model = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape, pooling=None)
    elif model_name == "ResNet50_imagenet":
        base_model = applications.ResNet50V2(include_top=False, input_shape=input_shape, pooling=None)
    elif model_name == "ResNet101":
        base_model = applications.ResNet101V2(include_top=False, weights=None, input_shape=input_shape, pooling=None)
    elif model_name == "ResNet152":
        base_model = applications.ResNet152V2(include_top=False, weights=None, input_shape=input_shape, pooling=None)

    """Get 7x7x2048 base model output"""
    x = base_model.output

    """Perform Pooling function to average each 7x7 matrix into 1 value = 2048 2D layer"""
    x = layers.GlobalAveragePooling2D()(x)

    """Perform Dimension Reduction by adding 1024 Hidden Layer"""
    x = layers.Dense(1024, activation='relu')(x)

    """Add Final Layer with Sigmoid activation function (since we are doing binary classification)"""
    if classes == 1:
        predictions = layers.Dense(1, activation='sigmoid')(x)
    else:
        predictions = layers.Dense(classes, kernel_initializer='normal', activation='softmax')(x)

    """Incase we want to perform transfer learning with only certain trainable layers layers
    N_Layers : Number of Trainable Layers"""
    if fine_tune:
        for i in range(len(base_model.layers) - N_layers):
            base_model.layers[i].trainable = False

    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model


def choose_basemodel3(input_shape1, classes=1, freeze=False, N=0,
                      model_name="local"):  # , input_shape3, input_shape4, classes=1, model_name="local"):

    """Initialize base model"""
    base_model1 = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape1,
                                          pooling='avg')  # , classes=classes)
    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model1.layers:
        layer._name = layer.name + str("_1")

    """Initialize base model"""
    base_model2 = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape1,
                                          pooling='avg')  # , classes=classes)
    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model2.layers:
        layer._name = layer.name + str("_2")

    """Initialize base model"""
    base_model3 = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape1,
                                          pooling='avg')  # , classes=classes)
    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model3.layers:
        layer._name = layer.name + str("_3")

    """Concatenate All the models together"""
    x = tf.keras.layers.concatenate([base_model1.output, base_model2.output, base_model3.output])

    """Perform Dimension Reduction by adding 1024 Hidden Layer"""
    x = layers.Dense(1024, activation='relu')(x)

    """Final Layer for Prediction"""
    predictions = layers.Dense(classes, activation='sigmoid')(x)

    model = models.Model(inputs=[base_model1.input, base_model2.input, base_model3.input], outputs=predictions)
    return model


######################## LSTM ##########################

class LSTMmodel(tf.keras.Model):
    def __init__(self, cnn_model, num_nodes, num_class):
        super(LSTMmodel, self).__init__()
        self.dense = layers.Dense(num_nodes, activation="relu")
        self.lstm = layers.LSTM(units=64, return_state=True, dropout=0.3)
        self.cnn_model = cnn_model
        self.binary = layers.Dense(num_class, activation="sigmoid")
        # output

    def call(self, input, **kwargs):
        CNN_model_distributed = layers.TimeDistributed(self.cnn_model)(input)
        x, state_h, state_c = self.lstm(CNN_model_distributed)
        dense = self.dense(x)
        output = self.binary(dense)

        LSTM_model = models.Model(inputs=input, outputs=output)

        return LSTM_model


# Alternatively:
def LSTM(cnn_model, input_shape, classes):
    """Time distribute CNN model"""
    input_layer = layers.Input(shape=input_shape)
    CNN_model_distributed = layers.TimeDistributed(cnn_model)(input_layer)

    # define the LSTM model
    x, state_h, state_c = layers.LSTM(units=64, return_state=True, dropout=0.3)(CNN_model_distributed)
    output = layers.Dense(classes, activation='softmax')
    LSTM_model = models.Model(inputs=input_layer, outputs=output)

    return None


def choose_multimodel2(path1, path2, classes=1, freeze_1=False, freeze_2=False, N=0):
    """Initialize pre-trained base model"""
    if path1 == "ResNet50":
        base_model1 = choose_model([500, 100, 1], classes=1)
    elif path1 == "VGG":
        base_model1 = choose_model([500, 100, 1], model_name="VGG16")
    elif path1 == "ResNet101":
        base_model1 = choose_model([500, 100, 1], model_name=path1)
    elif path1 == "ResNet152":
        base_model1 = choose_model([500, 100, 1], model_name=path1)
    else:
        base_model1 = models.load_model(path1, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model1.layers:
        layer._name += str("_A")

    # print(base_model1.summary())

    if freeze_1:
        for i in range(len(base_model1.layers) - N):
            base_model1.layers[i].trainable = False

    base_model1._layers.pop()  # Remove the output layer
    # base_model1._layers.pop()        # Remove the output layer

    """Initialize base model"""
    if path2 == "ResNet50":
        base_model2 = choose_model([500, 100, 1], classes=1)
    elif path2 == "VGG":
        base_model2 = choose_model([500, 100, 1], model_name="VGG16")
    elif path2 == "ResNet101":
        base_model2 = choose_model([500, 100, 1], model_name=path2)
    elif path2 == "ResNet152":
        base_model2 = choose_model([500, 100, 1], model_name=path2)
    else:
        base_model2 = models.load_model(path2, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model2.layers:
        layer._name += str("_B")
    # print(base_model2.summary())

    if freeze_2:
        for i in range(len(base_model2.layers) - N):
            base_model2.layers[i].trainable = False

    base_model2._layers.pop()  # Remove the output layer

    """Concatenate All the models together"""
    x = tf.keras.layers.concatenate([base_model1.layers[-1].output, base_model2.layers[-1].output])
    # x = tf.keras.layers.concatenate([base_model1.output, base_model2.output, base_model3.output])

    """Perform Dimension Reduction by adding 1024 Hidden Layer"""
    # x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Dense(3000, activation='relu')(x)
    x = layers.Dense(1000, activation='relu')(x)

    """Final Layer for Prediction"""
    if classes == 1:
        predictions = layers.Dense(classes, activation='sigmoid')(x)
    else:
        predictions = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=[base_model1.input, base_model2.input], outputs=predictions)
    return model


def choose_multimodel(input_shape, path1, path2, path3, classes=1, freeze_1=False, freeze_2=False, freeze_3=False, N=0):
    """Initialize pre-trained base model"""
    if path1 == "ResNet50":
        base_model1 = choose_model(input_shape, classes=1)
    elif path1 == "VGG":
        base_model1 = choose_model(input_shape, model_name="VGG16")
    elif path1 == "ResNet101":
        base_model1 = choose_model(input_shape, model_name=path1)
    elif path1 == "ResNet152":
        base_model1 = choose_model(input_shape, model_name=path1)
    else:
        base_model1 = models.load_model(path1, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model1.layers:
        layer._name += str("_A")

    # print(base_model1.summary())

    if freeze_1:
        for i in range(len(base_model1.layers) - N):
            base_model1.layers[i].trainable = False

    base_model1._layers.pop()  # Remove the output layer
    # base_model1._layers.pop()        # Remove the output layer

    """Initialize base model"""
    if path2 == "ResNet50":
        base_model2 = choose_model(input_shape, classes=1)
    elif path2 == "VGG":
        base_model2 = choose_model(input_shape, model_name="VGG16")
    elif path2 == "ResNet101":
        base_model2 = choose_model(input_shape, model_name=path2)
    elif path2 == "ResNet152":
        base_model2 = choose_model(input_shape, model_name=path2)
    else:
        base_model2 = models.load_model(path2, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model2.layers:
        layer._name += str("_B")
    # print(base_model2.summary())

    if freeze_2:
        for i in range(len(base_model2.layers) - N):
            base_model2.layers[i].trainable = False

    base_model2._layers.pop()  # Remove the output layer
    # base_model2._layers.pop()  # Remove the output layer

    # base_model2.outputs = [base_model2.layers[-2].output]

    """Initialize base model"""
    if path3 == "ResNet50":
        base_model3 = choose_model(input_shape, classes=1)
    elif path3 == "VGG":
        base_model3 = choose_model(input_shape, model_name="VGG16")
    elif path3 == "ResNet101":
        base_model3 = choose_model(input_shape, model_name=path3)
    elif path3 == "ResNet152":
        base_model3 = choose_model(input_shape, model_name=path3)
    else:
        base_model3 = models.load_model(path3, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model3.layers:
        layer._name += str("_C")

    # print(base_model3.summary())

    if freeze_3:
        for i in range(len(base_model3.layers) - N):
            base_model3.layers[i].trainable = False

    base_model3._layers.pop()  # Remove the output layer

    """Concatenate All the models together"""
    x = tf.keras.layers.concatenate(
        [base_model1.layers[-1].output, base_model2.layers[-1].output, base_model3.layers[-1].output])
    # x = tf.keras.layers.concatenate([base_model1.output, base_model2.output, base_model3.output])

    """Perform Dimension Reduction by adding 1024 Hidden Layer"""
    # x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Dense(3000, activation='relu')(x)
    x = layers.Dense(1536, activation='relu')(x)

    """Final Layer for Prediction"""
    if classes == 1:
        predictions = layers.Dense(classes, activation='sigmoid')(x)
    else:
        predictions = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=[base_model1.input, base_model2.input, base_model3.input], outputs=predictions)
    return model


def choose_multimodel4(path1, path2, path3, classes=1, freeze_1=False, freeze_2=False, freeze_3=False, N=0):
    """Initialize pre-trained base model"""
    if path1 == "ResNet50":
        base_model1 = choose_model([500, 100, 1], classes=1)
    elif path1 == "VGG":
        base_model1 = choose_model([500, 100, 1], model_name="VGG16")
    elif path1 == "ResNet101":
        base_model1 = choose_model([500, 100, 1], model_name=path1)
    elif path1 == "ResNet152":
        base_model1 = choose_model([500, 100, 1], model_name=path1)
    else:
        base_model1 = models.load_model(path1, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model1.layers:
        layer._name += str("_A")

    # print(base_model1.summary())

    if freeze_1:
        for i in range(len(base_model1.layers) - N):
            base_model1.layers[i].trainable = False

    #base_model1._layers.pop()  # Remove the output layer
    # base_model1._layers.pop()        # Remove the output layer

    """Initialize base model"""
    if path2 == "ResNet50":
        base_model2 = choose_model([500, 100, 1], classes=1)
    elif path2 == "VGG":
        base_model2 = choose_model([500, 100, 1], model_name="VGG16")
    elif path2 == "ResNet101":
        base_model2 = choose_model([500, 100, 1], model_name=path2)
    elif path2 == "ResNet152":
        base_model2 = choose_model([500, 100, 1], model_name=path2)
    else:
        base_model2 = models.load_model(path2, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model2.layers:
        layer._name += str("_B")
    # print(base_model2.summary())

    if freeze_2:
        for i in range(len(base_model2.layers) - N):
            base_model2.layers[i].trainable = False

    base_model2._layers.pop()  # Remove the output layer
    # base_model2._layers.pop()  # Remove the output layer

    # base_model2.outputs = [base_model2.layers[-2].output]

    """Initialize base model"""
    if path3 == "ResNet50":
        base_model3 = choose_model([500, 100, 1], classes=1)
    elif path3 == "VGG":
        base_model3 = choose_model([500, 100, 1], model_name="VGG16")
    elif path3 == "ResNet101":
        base_model3 = choose_model([500, 100, 1], model_name=path3)
    elif path3 == "ResNet152":
        base_model3 = choose_model([500, 100, 1], model_name=path3)
    else:
        base_model3 = models.load_model(path3, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model3.layers:
        layer._name += str("_C")

    # print(base_model3.summary())

    if freeze_3:
        for i in range(len(base_model3.layers) - N):
            base_model3.layers[i].trainable = False

    base_model3._layers.pop()  # Remove the output layer

    """Concatenate All the models together"""
    x = tf.keras.layers.concatenate(
        [base_model1.layers[-1].output, base_model2.layers[-1].output, base_model3.layers[-1].output])
    # x = tf.keras.layers.concatenate([base_model1.output, base_model2.output, base_model3.output])

    """Perform Dimension Reduction by adding 1024 Hidden Layer"""
    # x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Dense(3000, activation='relu')(x)
    x = layers.Dense(1536, activation='relu')(x)

    """Final Layer for Prediction"""
    if classes == 1:
        predictions = layers.Dense(classes, activation='sigmoid')(x)
    else:
        predictions = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=[base_model1.input, base_model2.input, base_model3.input], outputs=predictions)
    return model


def choose_multimodel3plus1(path1, path2, path3, classes=1, freeze_1=False, freeze_2=False, freeze_3=False, N=0):
    """Initialize pre-trained base model"""
    if path1 == "ResNet50":
        base_model1 = choose_model([500, 100, 1], classes=1)
    elif path1 == "VGG":
        base_model1 = choose_model([500, 100, 1], model_name="VGG16")
    elif path1 == "ResNet101":
        base_model1 = choose_model([500, 100, 1], model_name=path1)
    elif path1 == "ResNet152":
        base_model1 = choose_model([500, 100, 1], model_name=path1)
    else:
        base_model1 = models.load_model(path1, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model1.layers:
        layer._name += str("_A")

    # print(base_model1.summary())

    if freeze_1:
        for i in range(len(base_model1.layers) - N):
            base_model1.layers[i].trainable = False

    base_model1._layers.pop()  # Remove the output layer
    # base_model1._layers.pop()        # Remove the output layer

    """Initialize base model"""
    if path2 == "ResNet50":
        base_model2 = choose_model([500, 100, 1], classes=1)
    elif path2 == "VGG":
        base_model2 = choose_model([500, 100, 1], model_name="VGG16")
    elif path2 == "ResNet101":
        base_model2 = choose_model([500, 100, 1], model_name=path2)
    elif path2 == "ResNet152":
        base_model2 = choose_model([500, 100, 1], model_name=path2)
    else:
        base_model2 = models.load_model(path2, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model2.layers:
        layer._name += str("_B")
    # print(base_model2.summary())

    if freeze_2:
        for i in range(len(base_model2.layers) - N):
            base_model2.layers[i].trainable = False

    base_model2._layers.pop()  # Remove the output layer
    # base_model2._layers.pop()  # Remove the output layer

    # base_model2.outputs = [base_model2.layers[-2].output]

    """Initialize base model"""
    if path3 == "ResNet50":
        base_model3 = choose_model([500, 100, 1], classes=1)
    elif path3 == "VGG":
        base_model3 = choose_model([500, 100, 1], model_name="VGG16")
    elif path3 == "ResNet101":
        base_model3 = choose_model([500, 100, 1], model_name=path3)
    elif path3 == "ResNet152":
        base_model3 = choose_model([500, 100, 1], model_name=path3)
    else:
        base_model3 = models.load_model(path3, compile=False)

    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""
    for layer in base_model3.layers:
        layer._name += str("_C")

    # print(base_model3.summary())

    if freeze_3:
        for i in range(len(base_model3.layers) - N):
            base_model3.layers[i].trainable = False

    base_model3._layers.pop()  # Remove the output layer

    """This last Resnet will not be pre-trained"""

    base_model4 = choose_model([500, 100, 1], classes=1)
    base_model3._layers.pop()  # Remove the output layer

    """Concatenate All the models together"""
    x = tf.keras.layers.concatenate([base_model1.layers[-1].output, base_model2.layers[-1].output,
                                     base_model3.layers[-1].output, base_model4.layers[-1].output])
    # x = tf.keras.layers.concatenate([base_model1.output, base_model2.output, base_model3.output])

    """Perform Dimension Reduction by adding 1024 Hidden Layer"""
    # x = layers.GlobalAveragePooling2D()(x)

    # x = layers.Dense(3000, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)

    """Final Layer for Prediction"""
    if classes == 1:
        predictions = layers.Dense(classes, activation='sigmoid')(x)
    else:
        predictions = layers.Dense(classes, activation='softmax')(x)

    model = models.Model(inputs=[base_model1.input, base_model2.input, base_model3.input, base_model4.input],
                         outputs=predictions)
    return model


def base_multimodel(input_shape1, input_shape2,
                    classes=1):  # , input_shape3, input_shape4, classes=1, model_name="local"):

    """Initialize base model"""
    base_model1 = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape1,
                                          pooling='avg')  # , classes=classes)
    """To Avoid Name of Layers from the different ResNet50s being the same and TF throwing an error"""

    """Initialize base model"""
    base_model2 = applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape2,
                                          pooling='avg')  # , classes=classes)

    """Concatenate All the models together"""
    x = tf.keras.layers.concatenate(
        [base_model1.output, base_model2.output])  # , base_model3.output, base_model4.output])

    """Perform Dimension Reduction by adding 1024 Hidden Layer"""
    x = layers.Dense(1024, activation='relu')(x)

    """Final Layer for Prediction"""
    predictions = layers.Dense(classes, activation='sigmoid')(x)

    model = models.Model(inputs=[base_model1.input, base_model2.input], outputs=predictions)

    return model


def main():
    # LSTM(choose_model([100, 100, 1]), (100, 100, 1))
    from scipy import signal
    import matplotlib.pyplot as plt
    from pydub import AudioSegment
    rng = np.random.default_rng()

    from db_utils import plot_audio, plot_mfcc
    from librosa import stft

    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500 * np.cos(2 * np.pi * 25 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    noise = rng.normal(scale=np.sqrt(noise_power),
                       size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier + noise

    # file = "/Users/luisj/Downloads/frase1.wav"
    # sound_file = AudioSegment.from_wav(file)
    # mod = np.asarray(sound_file.get_array_of_samples())
    print(type(mod))
    # l = len(mod)
    # fs = sound_file.frame_rate
    #plot_audio(mod, fs)
    _, _, X = signal.spectrogram(mod, fs, window='hann', mode='magnitude')

    plot_mfcc(np.array(X), transpose=False)
    exit()

    #plt.show()
    return None


if __name__ == '__main__':
    main()
