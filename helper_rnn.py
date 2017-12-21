"""
Code reference: https://github.com/adeshpande3/LSTM-Sentiment-Analysis

"""
import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy.random import randint




def sep_line(words_file):
    """
    from a text file, return a list of sentence lines.
    Arguments: words_file (the text file filename)

    """
    ls = []
    for line in open(words_file, 'r'):
        ls.append(line)

    return ls

def extract_data(pos,neg,start,end):
    """
    Extract the sentences from the given list, return the list of sentences of length 2*(end-start) alongs with there labels.
    Arguments: pos (positive tweets)
               neg (negative tweets)
               start (start index)
               end (end index)
    """
    ls_pos = pos[start:end]
    ls_neg = neg[start:end]
    length = end-start
    labels = np.zeros((2*length,2), dtype=np.int32)
    labels[start:end] = [1,0]
    labels[end:-1] = [0,1]
    return (ls_pos + ls_neg, np.ndarray.tolist(labels))



def max_length_pad(ls):
    """
    Compute the length of the sentences with the highest number of words.
    Arguments : ls (the list of word-index encoded sentences)

    """
    res = -1
    for l in ls:
        res = max(res, len(l))

    return res


def generate_input(data,t):
    """
    Compute the index word vector for the given Data.
    Arguments:  data (list of sentences)
                t (The Keras instance that Tokenize the words)

    """
    encoded_docs = t.texts_to_sequences(data)
    max_length = max_length_pad(encoded_docs)

    return (encoded_docs,max_length)


def getTrainBatch(batchSize,maxSeqLength,padded_docs,nbr_tweet):
    """
    Return the subset of training data, the length defines by the batchsize.
    Arguments: batchSize (the size of the trainin subset)
               maxSeqLength (the length of the training element vector)
               padded_docs (The training data)
               nbr_tweet (The number of training data)

    """
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(0,nbr_tweet-1)
            labels.append([1,0])
        else:
            num = randint(nbr_tweet,2*nbr_tweet -1)
            labels.append([0,1])
        arr[i] = padded_docs[num]
    return arr, labels


def generate_test(filename):
    """
    From the text filename extract the id and the  tweet.
    Arguments:  filename (the name of the text file)

    """
    test_data = open(filename, "r", encoding='utf-8')
    id_ = [line[:line.find(",")] for line in test_data.readlines()]

    test_data = open(filename, "r", encoding='utf-8')
    test = [line[line.find(",") + 1:].strip() for line in test_data.readlines()]

    return (id_,test)

