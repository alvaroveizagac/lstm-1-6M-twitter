from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano


def prepare_data(seqs, labels, maxlen=None):
    """
    x -> seqs, y -> labels 
    Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    # it will have the lenghts of the seqs   each comment has a vector and the len it is the size of the vector 
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None
    # n_samples = number of the sequences 
    n_samples = len(seqs)
    # maxeln the max lenghts of the sequences
    maxlen = numpy.max(lengths)
    # numpy.zeros Return a new array of given shape and type, filled with zeros.
    # x = 0 
    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    # x = 0.0  x (length) -> maxlen,  x[0](length) --> n_samples   
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_dataset_file(dataset, default_dataset, origin):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present
    os.path.isfile("C:/Users/AlvaroVeizaga/Anaconda2/Lib/alvaro.txt")  ---> Return: True

    '''
    data_dir, data_file = os.path.split(dataset)
    
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path
            
    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Downloading data from %s' % origin)
        print("alvaro_entro 0.5")
        urllib.request.urlretrieve(origin, dataset)
        print("alvaro_entro 1")

    
    print("alvaro_entro 2")        
    return dataset

# for the example using the imdb the path was path="imdb.pkl"
def load_data(path="/resources/data/twitter_1_6M/twitter_1_6M.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    # original file was path imdb.pkl and the url was = http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl
    path = get_dataset_file(
        path, "twitter_1_6M.pkl",
        "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.phkl")
    print("alvaro_entro 0")
    print (path)

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    # load from the dataset we have different train and test each has 25000 (50% pos and 50% neg)
    train_set = pickle.load(f)
    test_set = pickle.load(f)

    f.close()

    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    # train_set has 2 dimensions [0] and [1] each length 25000
    # inside dimension [0] each of the 25000 has another content 
    # inside dimension [1] each of the 25000 has wether 0 or 1 
    # [0][24999] -> an array of 200 elements  and  [1][24999]  -> only a int "0"
    train_set_x, train_set_y = train_set # _x -> [0]  and _y -> [1]
    n_samples = len(train_set_x) #25000 
    sidx = numpy.random.permutation(n_samples) # mix all the values np.random.permutation([1, 4, 9, 12, 15])  Output: (12, 4, 15, 1, 9)
    #n_samples values from 0 to 25000 are split and each value separated
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    #n_samples = 25000  0.1% -> 2500  -->  n_train = 22500
    #valid_set_x and valid_set_y  length is 2500  
    #creating both arrays dinamically and randomly 
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    #train_set_x and train_set_y  lenght is 22500  both creating dinamically and random using sidx
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    #tuples
    #train_set length  22500  [0] ->  each of the 22500 values has a array of diff values [1]-> 0 or 1 values
    train_set = (train_set_x, train_set_y)
    #valid_set  length 2500 [0] ->  each of the 22500 values has a array of diff values [1]-> 0 or 1 values
    valid_set = (valid_set_x, valid_set_y)
    #n_words=100000
    # if w >= n_words set to (1) otherwise return the original words number.
    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]
    #test set is split    
    test_set_x, test_set_y = test_set #25000
    #train set is split   into valid (2500) and train set (22500)  total 25000
    valid_set_x, valid_set_y = valid_set #2500
    train_set_x, train_set_y = train_set #22500

    train_set_x = remove_unk(train_set_x) 
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    #Sort the values od the data structure 
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    # Sort by the sequence of lenght, sorting based on the number of words 
    # from the lowest number of words to the highest
    # valid_set_x -> [0]  -> length 32, [1] -> length 36, [2] -> 39, [2490] -> 1228, [2499] -> 1443    
    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y) # tuple the values and the labels  22500
    valid = (valid_set_x, valid_set_y) # tuple values and the labels  2500
    test = (test_set_x, test_set_y) # tuple values and the labels 25000
    #return the dataset explained above 
    return train, valid, test
