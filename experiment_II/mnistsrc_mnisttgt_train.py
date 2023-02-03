import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
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

## Stage 1 model name
src_model_name = "mnistsrc"

## Stage 1 model file path
src_model_path = "models/"+src_model_name+".mdl"

## Stage 2 model name
tgt_model_name = src_model_name+"_"+"mnisttgt"

## Stage 2 model file path
tgt_model_path = "models/"+tgt_model_name+".mdl"

## Stage 2 model evaluation history file path
tgt_model_hist = "models/"+tgt_model_name+".xlsx"

## Pulling the trained Stage 1 model
tgt_model = tf.keras.models.load_model(src_model_path)

batch_size = 64
num_classes = 10
num_epochs = 10

## input image dimensions
img_rows, img_cols = 28, 28

## Data source
data_source = "local_dir" # "keras_data_set" or "local_dir"

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

tgt_model.summary()

## Train the model
tgt_model.fit\
(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
    verbose=1
)

tgt_model.save(tgt_model_path, save_format='h5')

score = tgt_model.evaluate(x_test, y_test, verbose=0)

print('Test loss: %.30f'%score[0])
print('Test accuracy: %.30f'%score[1])

## Store training output
metrics_history = pd.DataFrame.from_dict(tgt_model.history.history)
metrics_history = pd.concat([metrics_history['loss'], metrics_history['accuracy'], metrics_history['val_loss'], metrics_history['val_accuracy']], axis=1)
metrics_history.T.to_excel(tgt_model_hist)