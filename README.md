# Undergraduate Thesis Code

This repository contains all the Python source code files associated with the experiments conducted for my undergraduate thesis with title *Applications of Convolutional Neural Networks in Image and Natural Language Processing*. To understand the purpose of these experiments feel free to read the *Hypothesis and Methodology* and *Experimental Evaluation* chapters of [[my thesis][thesis_text]]. The experiment source code files are trainers of the supervised Deep Learning models called Convolutional Neural Networks (CNNs), which are used to solve the image classification task of handwritten digits from 0 to 9 (10 classes), and the task of classifying an IMDb review in text, as either a positive or a negative review (2 classes).

The LMRDv1.0 dataset (or the IMDb dataset) corresponds to the IMDb movie review dataset and a slightly modified version of this dataset was pulled from [[IMDb Dataset][imdb_dataset_repo]]. The MNIST dataset corresponds to the handwritten digit image classification task and was pulled from [[MNIST Dataset][mnist_dataset_repo]]. The respective script files specifying the preprocessing and training configuration for the CNN trainers on the MNIST and IMDb datasets, were pulled from [[MNIST Code][mnist_code_repo]] and [[IMDb Code][imdb_code_repo]] respectively, where each contains its own CNN architecture which is tuned for the respective classification task.

## Files and File Hierarchy

The directory &lt;./experiment_I> contains the script files corresponding to the first set of experiments (subchapter *Experiments Part I*) and &lt;./experiment_II> contains the script files corresponding to the second set of experiments (subchapter *Experiments Part II*). The &lt;./datasets> directory contains the image and text datasets on their respective &lt;./datasets/MNIST> and &lt;./datasets/IMDb> directories. To run the experiment executable files, change the directory to the respective experiment directory. 

The &lt;data_helpers.py> file contains the necessary Python functions that are used by the experiment executable files to parse and preprocess the MNIST and IMDb datasets. The &lt;w2v<span></span>.py> file contains the necessary Python function named as &lt;train_word2vec> that is used by the experiment executable files to train and store word2vec models. All of the trained models and their respective diagrams are saved under the &lt;models> directories.

### Experiments Part I

The directories &lt;./experiment_I/python2>, &lt;./experiment_I/python3> contain the script models and code for the first set of experiments named *Experiments Part I*. The code was written in Python 2 and Python 3 respectively. Each of the &lt;./experiment_I/python2>, &lt;./experiment_I/python3> directories, contain a &lt;arch_opt4_mnist_dataset_mnist.py>, &lt;arch_opt4_mnist_dataset_imdb.py>, &lt;arch_opt4_imdb_dataset_imdb.py> and a &lt;arch_opt4_imdb_dataset_imdb.py> source code file. These source code files are all CNN classifier trainers that can train CNN classifiers either on image or text labeled datasets. An architecture is specified in each of these source code files, and we refer to the &lt;&lt;arch_name>.py> file's architecture name as &lt;arch_name> (e.g. the file &lt;arch_opt4_mnist_dataset_mnist.py> corresponds to architecture &lt;arch_opt4_mnist_dataset_mnist>).
\
Executables:
- &lt;arch_opt4_mnist_dataset_mnist.py>: Trains a CNN classifier on the MNIST dataset. The training configuration has been taken from [[MNIST Code][mnist_code_repo]]. The &lt;arch_opt4_mnist_dataset_mnist> architecture is tuned for the image classification task corresponding to the MNIST dataset and the respective executable is used to train on the MNIST dataset.
- &lt;arch_opt4_imdb_dataset_imdb.py>: Trains a CNN classifier on the IMDb dataset. The training configuration has been taken from [[IMDb Code][imdb_code_repo]]. The &lt;arch_opt4_imdb_dataset_imdb> architecture is tuned for the text classification task corresponding to the IMDb dataset and the respective executable is used to train on the IMDb dataset.
- &lt;arch_opt4_mnist_dataset_imdb.py>: Trains a CNN classifier on the IMDb dataset. The architecture &lt;arch_opt4_mnist_dataset_imdb> is the same as the architecture &lt;arch_opt4_mnist_dataset_mnist> with the only difference being that the 2 dimensional layers were changed to be 1 dimensional (e.g. each 2D convolutional layer became a 1D convolutional layer).
- &lt;arch_opt4_imdb_dataset_mnist.py>: Trains a CNN classifier on the MNIST dataset. The architecture &lt;arch_opt4_mnist_dataset_imdb> is the same as the architecture &lt;arch_opt4_mnist_dataset_mnist> with the only difference being that the 1 dimensional layers were changed to be 2 dimensional (e.g. each 2D convolutional layer became a 1D convolutional layer).

&lt;./experiment_I/data_helpers.py> contains the necessary functions that are used by the executables to parse and preprocess the MNIST and IMDb datasets. &lt;./experiment_I/w2v<span></span>.py> contains the function that trains and stores word2vec models. All of the trained models and their respective diagrams are saved under the &lt;models> directory.

### Language and library versions used

For reproducible and stable trainings, the versions used for these experiments are
- Python 2 with Keras: 2.3.0 and Tensorflow: 1.13.1
- Python 3 with Keras: 2.3.0 and Tensorflow: 2.0.0

## Experiments Part II (Using Transfer Learning)

The directory &lt;./experiment_II> contain the script models and code for the second set of experiments named *Experiments Part II*. The code was written in Python 3. The files &lt;mnistsrc_train.py>, &lt;imdbsrc_train.py>, &lt;mnistsrc_mnisttgt_train.py>, &lt;mnistsrc_imdbtgt_train.py>, &lt;imdbsrc_imdbtgt.py> and &lt;imdbsrc_mnisttgt.py> are the experiment source code files. These source code files are all CNN classifier trainers that can train CNN classifiers on image and text labeled datasets. An architecture is specified in each of these source code files, and we refer to the &lt;&lt;arch_name>_train.py> file's architecture name as &lt;arch_name>. As Transfer Learning is implemented in this part of experiments, there needs to be 2 stages of training in total. The pre-training stage and the fine-tuning stage. The pre-training is done by the &lt;./experiment_II/mnistsrc_train.py> and &lt;./experiment_II/imdbsrc_train.py>, and the fine-tuning is done by &lt;mnistsrc_mnisttgt_train.py>, &lt;mnistsrc_imdbtgt_train.py>, &lt;imdbsrc_imdbtgt.py> and &lt;imdbsrc_mnisttgt.py>.

- The &lt;./experiment_II/mnistsrc_train.py> file contains the executable code that trains a CNN classifier based on [[MNIST Code][mnist_code_repo]], on the MNIST dataset. It randomly initializes all the learnable parameters. The model produced by this training is saved as &lt;./experiment_II/models/mnistsrc.mdl>. The &lt;mnistsrc> architecture is tuned for the image classification task corresponding to the MNIST dataset and the respective executable is used to train a source model on the MNIST dataset.
- The &lt;./experiment_II/imdbsrc_train.py> file contains the executable code that trains a CNN classifier based on [[IMDb Code][imdb_code_repo]], on the IMDb dataset. It randomly initializes all the learnable parameters. The model produced by this training is saved as &lt;./experiment_II/models/imdbsrc.mdl>. Additionally it uses a word2vec trainer to refine the initial embedding layer learnable parameters. The &lt;imdbsrc> architecture is tuned for the text classification task corresponding to the IMDb dataset and the respective executable is used to train a source model on the IMDb dataset.
- The &lt;./experiment_II/mnistsrc_mnisttgt_train.py> file contains the executable code that trains a CNN classifier whose hidden layers and its learnable parameters are copied from &lt;./experiment_II/models/mnistsrc.mdl>. It is trained on the MNIST dataset.
- The &lt;./experiment_II/mnistsrc_imdbtgt_train.py> file contains the executable code that trains a CNN classifier whose hidden layers and its learnable parameters are copied from &lt;./experiment_II/models/mnistsrc.mdl> with a minor exception, an additional appropriate embedding layer is added next to the input layer to receive embedded representations of words from  the input text. It is trained on the IMDb dataset.
- The &lt;./experiment_II/imdbsrc_imdbtgt_train.py> file contains the executable code that trains a CNN classifier whose hidden layers and its learnable parameters are copied from &lt;./experiment_II/models/imdbsrc.mdl> . It is trained on the IMDb dataset.
- The &lt;./experiment_II/imdbsrc_mnisttgt_train.py> file contains the executable code that trains a CNN classifier whose hidden layers and its learnable parameters are copied from &lt;./experiment_II/models/imdbsrc.mdl> with a minor exception, the embedding layer next to the input layer is removed. It is trained on the MNIST dataset.

&lt;./experiment_II/data_helpers.py> contains the necessary functions that are used by the executables to parse and preprocess the MNIST and IMDb datasets. &lt;./experiment_II/w2v<span></span>.py> contains the function that trains and stores word2vec models. All of the trained models and their respective diagrams are saved under the &lt;models> directory.

### Language and library versions used

For reproducible and stable trainings, the versions used for these experiments are
- Python 3 with Keras: 2.3.0 and Tensorflow: 2.0.0


[thesis_text]: <https://hellanicus.lib.aegean.gr/handle/11610/23105>
[mnist_code_repo]: <https://github.com/keras-team/keras/blob/1a3ee8441933fc007be6b2beb47af67998d50737/examples/mnist_cnn.py>
[mnist_dataset_repo]: <http://yann.lecun.com/exdb/mnist/>
[imdb_code_repo]: <https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras>
[imdb_dataset_repo]: <https://github.com/linanqiu/word2vec-sentiments>