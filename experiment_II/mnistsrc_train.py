import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import data_helpers


## ! Unnecessary stdout suppression: Begin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging; tf.get_logger().setLevel(logging.ERROR)

## ! Unnecessary stdout suppression: End

## ! RNG seeds: Begin

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

## ! RNG seeds: End

## ! Configuration: Begin

batch_size = 64
num_classes = 10
num_epochs = 10

## input image dimensions
img_rows, img_cols = 28, 28

## Data source
data_source = "local_dir" # "keras_data_set" or "local_dir"

## Output model's name
src_model_name = "mnistsrc"

## Output model's file path
src_model_path = "models/"+src_model_name+".mdl"

## Output model's evaluation history file path
src_model_hist = "models/"+src_model_name+".xlsx"

## ! Configuration: End

def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = \
        data_helpers.load_local_mnist_dataset\
        (
            num_classes=num_classes,
            img_rows=img_rows,
            img_cols=img_cols
        )

    return x_train, y_train, x_test, y_test

## Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test = load_data(data_source)
# x_train, y_train, x_test, y_test = x_train[:5000], y_train[:5000], x_test[:5000], y_test[:5000] # ! DEBUG SETTING

input_shape = x_train.shape[1:]

print('Input shape: ' + str(input_shape))

src_model = Sequential()
src_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
src_model.add(Conv2D(64, (3, 3), activation='relu'))
src_model.add(MaxPooling2D(pool_size=(2, 2)))
src_model.add(Dropout(0.25))
src_model.add(Flatten())
src_model.add(Dense(128, activation='relu'))
src_model.add(Dropout(0.5))
src_model.add(Dense(num_classes, activation='softmax'))

src_model.compile\
(
    loss=categorical_crossentropy,
    optimizer=Adadelta(),
    metrics=["accuracy"]
)

src_model.summary()

## Train the model
src_model.fit\
(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
    verbose=1
)

src_model.save(src_model_path)

score = src_model.evaluate(x_test, y_test, verbose=0)

print('Test loss: %.30f'%score[0])
print('Test accuracy: %.30f'%score[1])

## Store training output
metrics_history = pd.DataFrame.from_dict(src_model.history.history)
metrics_history = pd.concat([metrics_history['loss'], metrics_history['accuracy'], metrics_history['val_loss'], metrics_history['val_accuracy']], axis=1)
metrics_history.T.to_excel(src_model_hist)