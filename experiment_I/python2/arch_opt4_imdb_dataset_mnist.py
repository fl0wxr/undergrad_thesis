import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import data_helpers


## ! Unnecessary stdout suppression: Begin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

## ! Unnecessary stdout suppression: End

## ! RNG seeds: Begin

random.seed(1)
np.random.seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
tf.set_random_seed(1)

## ! RNG seeds: End

## ! Configuration: Begin

## Output model
model_name = 'arch_opt4_imdb_dataset_mnist'
model_path = './models/'+model_name

num_classes = 10

## input image dimensions
img_rows, img_cols = 28, 28

## Model Hyperparameters
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

## Training parameters
batch_size = 64
num_epochs = 10

## Data source
data_source = "local_dir"  # "keras_data_set" or "local_dir"

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

# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test = load_data(data_source)
# x_train, y_train, x_test, y_test = x_train[:5000], y_train[:5000], x_test[:5000], y_test[:5000]

## ! Build model: Begin

input_shape = x_train.shape[1:]

print('Input shape: ' + str(input_shape))

model_input = Input(shape=input_shape)
z = Dropout(dropout_prob[0])(model_input)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = \
    Conv2D\
    (
        filters=num_filters,
        kernel_size=sz,
        padding="valid",
        activation="relu",
        strides=1
    )(z)
    conv = MaxPooling2D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(num_classes, activation='softmax')(z)

model = Model(model_input, model_output)

## ! Build model: End

model.compile\
(
    loss=categorical_crossentropy,
    optimizer=Adadelta(),
    metrics=["accuracy"]
)

model.summary()

# Train the model
model.fit\
(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
    verbose=1
)

model.save(''.join((model_path, '.mdl')))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss: %.30f'%score[0])
print('Test accuracy: %.30f'%score[1])

## Store training output
metrics_history = pd.DataFrame.from_dict(model.history.history)
metrics_history = pd.concat([metrics_history['loss'], metrics_history['accuracy'], metrics_history['val_loss'], metrics_history['val_accuracy']], axis=1)
metrics_history.T.to_excel(''.join((model_path, '.xlsx')))