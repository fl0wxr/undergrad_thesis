import io
import re
import itertools
from collections import Counter
import idx2numpy
import numpy as np
from tensorflow.python.keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical


## ! Image processing tools: Begin

def load_preprocessed_keras_mnist_dataset(num_classes, img_rows, img_cols):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = mnist_channel_axis_add(x=np.concatenate((x_train, x_test), axis=0), img_rows=img_rows, img_cols=img_cols)
    y = np.concatenate((y_train, y_test), axis=0)

    (x_train, y_train), (x_test, y_test) = preprocess_image_dataset(x, y, num_classes=num_classes)

    return (x_train, y_train), (x_test, y_test)

def load_local_mnist_dataset(num_classes, img_rows, img_cols):
    '''
    Description:
        Loads the MNIST dataset from a local directory.
    
    Inputs:
        <num_classes>: int, total number of classes for the classification task.
        <img_rows>: int, total number of the feature's image matrix rows.
        <img_cols>: int, total number of the feature's image matrix columns.

    Returns:
        <dataset>: tuple, contains 2 tuples each containing 2 elements. It is the dataset after it has been preprocessed. <dataset> = (<x_train>, <y_train>), (<x_test>, <y_test>):
            <x_train>: numpy.ndarray, the training set's feature tensor, has shape (<predefined_number_of_training_instances>, <predefined_number_of_rows>, <predefined_number_of_cols>, 1).
            <y_train>: numpy.ndarray, ground truth of the test set with shape (<predefined_number_of_training_instances>, <predefined_number_of_classes>).
            <x_test>: numpy.ndarray, the test set's feature tensor, has shape (<predefined_number_of_test_instances>, <predefined_number_of_feature_rows>, <predefined_number_of_feature_cols>, 1).
            <y_test>: numpy.ndarray, ground truth of the test set with shape (<predefined_number_of_test_instances>, <predefined_number_of_classes>).
    '''

    rel_path_dir = '../../datasets/MNIST'
    rel_path_x_train = rel_path_dir+'/train/train-images-idx3-ubyte'
    rel_path_y_train = rel_path_dir+'/train/train-labels-idx1-ubyte'
    rel_path_x_test = rel_path_dir+'/test/t10k-images-idx3-ubyte'
    rel_path_y_test = rel_path_dir+'/test/t10k-labels-idx1-ubyte'

    x_train = idx2numpy.convert_from_file(rel_path_x_train)
    y_train = idx2numpy.convert_from_file(rel_path_y_train)
    x_test = idx2numpy.convert_from_file(rel_path_x_test)
    y_test = idx2numpy.convert_from_file(rel_path_y_test)

    x = mnist_channel_axis_add(x=np.concatenate((x_train, x_test), axis=0), img_rows=img_rows, img_cols=img_rows)
    y = np.concatenate((y_train, y_test), axis=0)

    (x_train, y_train), (x_test, y_test) = preprocess_image_dataset(x, y, num_classes=num_classes)

    return (x_train, y_train), (x_test, y_test)

def preprocess_image_dataset(x, y, num_classes):
    '''
    Description:
        Main preprocessing of the given dataset.

    Inputs:
        <x>: numpy.ndarray, the dataset's feature tensor. Has shape (<number_of_instances>, <number_of_feature_rows>, <number_of_feature_cols>, 1).
        <y>: numpy.ndarray, the dataset's ground truth tensor. Has shape (<number_of_instances>, <number_of_classes>).
        <num_classes>: int, total number of classes.

    Returns:
        <dataset>: tuple, 2 tuples each containing 2 elements. It is the dataset after it has been preprocessed. <dataset> = (<x_train>, <y_train>), (<x_test>, <y_test>):
            <x_train>: numpy.ndarray, the training set's feature tensor, has shape (<predefined_number_of_training_instances>, <predefined_number_of_rows>, <predefined_number_of_cols>, 1).
            <y_train>: numpy.ndarray, ground truth of the test set in OHE with shape (<predefined_number_of_training_instances>, <predefined_number_of_classes>).
            <x_test>: numpy.ndarray, the test set's feature tensor, has shape (<predefined_number_of_test_instances>, <predefined_number_of_feature_rows>, <predefined_number_of_feature_cols>, 1).
            <y_test>: numpy.ndarray, ground truth of the test set in OHE with shape (<predefined_number_of_test_instances>, <predefined_number_of_classes>).
    '''

    ## Dataset format modifier
    x.astype('float32')

    ## Feature normalizer
    x = x/255.

    ## One hot encoding for classes
    y = to_categorical(y, num_classes)

    ## Dataset shuffler
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]

    ## Dataset splitter
    train_len = int(len(y) * 6./7.)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return (x_train, y_train), (x_test, y_test)

def mnist_channel_axis_add(x, img_rows, img_cols):
    '''
    Description:
        Adds another axis to the input, and returns that expanded input.

    Inputs:
        <x>: numpy.ndarray, the dataset's feature tensor in which there is going to be assigned an additional axis to. Has shape (<number_of_instances>, <number_of_feature_rows>, <number_of_feature_cols>, 1).
        <img_rows>: int, total number of the features' image matrix rows.
        <img_cols>: int, total number of the features' image matrix columns.

    Returns:
        <x_expanded_axis>: numpy.ndarray, the input tensor with an additional axis.
    '''

    if K.image_data_format() == 'channels_first':
        print('Warning: Channels axis is before rows and columns.')
        x = x.reshape(x.shape[0], 1, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 1)

    return x

## ! Image processing tools: End

## ! NL processing tools: Begin

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Description:
        Tokenization/string cleaning for all datasets except for SST. Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    Inputs:
        <string>: str.

    Returns:
        <string_cleaned>: str.
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

def load_imdb_data_and_labels():
    """
    Description:
        Loads MR polarity data from files, splits the data into words and generates labels. Returns split sentences and labels.

    Returns:
        <dataset>: list, contains 2 elements. <dataset> = [<x_text>, <y>]:
            <x_text>: list, is consisted of all the dataset's features. Contains <number_of_instances> elements which are all lists, where each of these lists is a feature instance that contains token (word) strings. These are splitted tokens from the text corresponding to that particular instance.
            <y>: numpy.ndarray, contains the dataset's ground truth. It has shape (<number_of_instances>).
    """

    ## Load data from files
    positive_training_examples = io.open("../../datasets/IMDb/train-pos.txt", encoding='iso-8859-1').readlines()
    positive_training_examples = [s.strip() for s in positive_training_examples]
    negative_training_examples = io.open("../../datasets/IMDb/train-neg.txt", encoding='iso-8859-1').readlines()
    negative_training_examples = [s.strip() for s in negative_training_examples]

    positive_testing_examples = io.open("../../datasets/IMDb/test-pos.txt", encoding='iso-8859-1').readlines()
    positive_testing_examples = [s.strip() for s in positive_testing_examples]
    negative_testing_examples = io.open("../../datasets/IMDb/test-neg.txt", encoding='iso-8859-1').readlines()
    negative_testing_examples = [s.strip() for s in negative_testing_examples]

    ## ! Dataset concat: Begin

    positive_examples = positive_training_examples + positive_testing_examples
    negative_examples = negative_training_examples + negative_testing_examples

    ## Split by words
    x_text = positive_examples + negative_examples

    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    ## Generate labels
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]

    y = np.concatenate([positive_labels, negative_labels], 0)

    ## ! Dataset concat: End

    return [x_text, y]

def pad_sentences(sentences, maxlen=-1, padding_word="<PAD/>"):
    """
    Description:
        Pads all sentences to the same length which is set to be equal to the maximum number of tokens among all instance texts. The length is defined by the longest sentence and returns padded sentences.

    Inputs:
        <sentences>: list, is consisted of all the dataset's features. Contains <number_of_instances> elements which are all lists, where each of these lists is a feature instance that contains token (word) strings. These are splitted tokens from the text corresponding to that particular instance.
        <maxlen>: int, default value: -1, after each text instance has been padded, -(<maxlen>+1) tokens from the right towards the left are dropped.
        <padding_word>: str, default value: '<PAD/>', this is the padding token.

    Returns:
        <padded_reduced_sentences>: list, contains the same text elements as <sentences> but each text element is now padded and trimmed from the right so that all text elements share the same number of token elements, which is set to be equal to <maxlen>.
    """

    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    ## A sentence has length equal to the length of the sentence with the greatest length of <sentence>, and the next line shrinks this length to <maxlen> size.

    padded_reduced_sentences = \
        [
            padded_sentences[sentence_idx][:maxlen]
            for sentence_idx in range(len(padded_sentences))
        ]

    return padded_reduced_sentences

def build_vocab(sentences):
    """
    Description:
        Builds a vocabulary mapping from word to index based on the sentences and returns vocabulary mapping and inverse vocabulary mapping.

    Inputs:
        <sentences>: list, contains the text features. All the text features share the same length <maxlen>. Also note that some of the raw feature set's tokens may be missing.

    Returns:
        <vocabulary_pair>: list, contains 2 elements, <vocabulary_pair> = [<vocabulary>, <vocabulary_inv>]:
            <vocabulary>: dict, each key is the vocabularies token in str type and each value is the unique token's index which is an int type, beginning from 0 and ending to the total number of unique tokens found inside the feature set, each time incrementing the key value by 1. The zeroeth value contains the padding token, and beginning from key with value 1 until the maximum dictionary key value, the tokens are sorted in descending order by the frequency of their token appearances inside the feature set.
            <vocabulary_inv>: list, each element is a vocabularies token in str. Each list index versus string word pair is the same as in <vocabulary>. It can be seen as the inverse of <vocabulary> in regards to its key-value pair.
    """

    ## Counts the number of encounters of each word on the dataset, it resembles a dictionary where you input the word, and it returns it's number of encounters.
    word_counts = Counter(itertools.chain(*sentences))
    ## Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    ## Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Description:
        Based on a vocabulary, the tokens are substituted by their respective vocabulary indices inside the feature set, and because text instances have equal size, the feature list can be converted into a feature tensor, which is invoked here.

    Inputs:
        <sentences>: list, contains the text features. All the text features share the same length <maxlen>.
        <labels>: numpy.ndarray, the ground truth vector.

    Return:
        <dataset>: list, consisted of 2 elements. <dataset> = [x, y]:
            <x>: numpy.ndarray, the feature matrix, has shape (<number_of_instances>, <maxlen>).
            <y>: numpy.ndarray, the ground truth vector, has shape (<number_of_instances>).
    """

    ## Converts a given sentence, which is a list consisted of string elements, to their respective vocabulary indices (0 is the padding or <PAD/>)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)

    return [x, y]

def load_local_imdb_dataset(sequence_length):
    """
    Description:
        Loads and preprocesses data for the MR dataset. Returns input vectors, labels, vocabulary, and the inverse vocabulary.

    Inputs:
        <sequence_length>: int, the resulting feature instances are all going to have an equal <sequence_length> length. This is the sequence length.

    Returns:
        <dataset_vocabulary>: list, contains 3 elements where the first two elements are tuples each composed by 2 elements. <dataset_vocabulary> = [(<x_train>, <y_train>), (<x_test>, <y_test>), <vocabulary_inv>]:
            <x_train>: numpy.ndarray, the training feature matrix with shape (<number_of_training_instances>, <sequence_length>).
            <y_train>: numpy.ndarray, the training ground truth vector with shape (<number_of_training_instances>).
            <x_test>: numpy.ndarray, the test feature matrix with shape (<number_of_test_instances>, <sequence_length>).
            <y_test>: numpy.ndarray, the test ground truth vector with shape (<number_of_test_instances>).
            <vocabulary_inv>: dict, the inverted vocabulary defined as per the function <build_vocab>.
    """

    # Load and preprocess data
    x, y = load_imdb_data_and_labels()

    x = pad_sentences(x, maxlen=sequence_length)

    vocabulary, vocabulary_inv = build_vocab(x)

    ## Converting into a dictionary instead of a list
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv)}

    x, y = build_input_data(x, y, vocabulary)

    ## Shuffle data
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.5)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return [(x_train, y_train), (x_test, y_test), vocabulary_inv]

def batch_iter(data, batch_size, num_epochs):
    """
    Description:
        Generates a batch iterator for a dataset.
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

## ! NL processing tools: End