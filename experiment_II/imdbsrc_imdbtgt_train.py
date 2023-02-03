from copy import deepcopy
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import data_helpers
from keras.datasets import imdb
from keras.preprocessing import sequence
from w2v import train_word2vec


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
src_model_name = "imdbsrc"

## Stage 1 model file path
src_model_path = "models/"+src_model_name+".mdl"

## Stage 2 model name
tgt_model_name = src_model_name+"_"+"imdbtgt"

## Stage 2 model file path
tgt_model_path = "models/"+tgt_model_name+".mdl"

## Stage 2 model evaluation history file path
tgt_model_hist = "models/"+tgt_model_name+".xlsx"

## Pulling the trained Stage 1 model
src_model = tgt_model = tf.keras.models.load_model(src_model_path)

## The Stage 1 input and embedding layers' hyperparameters
src_embedding_dim, src_sequence_length = tuple(src_model.layers[1].output.shape[1:])

## Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand or CNN-non-static or CNN-static

## Data source
data_source = "local_dir"  # keras_data_set or local_dir

padding_word = "<PAD/>"

## Model Hyperparameters
embedding_dim = src_embedding_dim
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

## Training parameters
batch_size = 64
num_epochs = 10

## Prepossessing parameters
sequence_length = src_sequence_length
max_words = 5000

## Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

## ! Configuration: End

def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        print('Using the keras IMDb dataset')
        (x_train, y_train), (x_test, y_test) = \
        imdb.load_data\
        (
            num_words=max_words,
            start_char=None,
            oov_char=None,
            index_from=None
        )

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = deepcopy(padding_word)
    else:
        print('Using a local IMDb dataset.')
        (x_train, y_train), (x_test, y_test), vocabulary_inv = data_helpers.load_local_imdb_dataset(sequence_length)

    return x_train, y_train, x_test, y_test, vocabulary_inv


## Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data(data_source)
# x_train, y_train, x_test, y_test = x_train[:10000], y_train[:10000], x_test[:10000], y_test[:10000] # ! DEBUG SETTING

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("x_train shape: " + str(x_train.shape))
print("x_test shape: " + str(x_test.shape))
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

tgt_model.summary()

## Train the model
tgt_model.fit\
(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
    verbose=1,
)

tgt_model.save(tgt_model_path, save_format='h5')

score = tgt_model.evaluate(x_test, y_test, verbose=0)

print('Test loss: %.30f'%score[0])
print('Test accuracy: %.30f'%score[1])

## Store training output
metrics_history = pd.DataFrame.from_dict(tgt_model.history.history)
metrics_history = pd.concat([metrics_history['loss'], metrics_history['accuracy'], metrics_history['val_loss'], metrics_history['val_accuracy']], axis=1)
metrics_history.T.to_excel(tgt_model_hist)