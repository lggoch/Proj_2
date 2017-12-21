import numpy as np
from multiprocessing import Pool
from sent2vec import get_sentence_embeddings

global word_vector

def avg_words(words):
    """
    Take the words of a tweet, get their embeddings and then compute the average vector between all embeddings
    :param words: an array of all words that a tweet contains
    :return: the average of all words that represent the embedding the the tweet
    """
    global word_vector
    try:
        word_emb = word_vector.loc[words]
        word_emb = word_emb.dropna(axis=0)
        word_emb = word_emb.astype("float64")
        return word_emb.mean(0)
    except KeyError:
        return np.zeros(word_vector.shape[1])


def sentence_embedding(train_pos_filename, train_neg_filename, test_data_filename):
    """
    Embedding of all the data using the sent2vec algorithm
    :param train_pos_filename: The path of the training data containing positive tweets
    :param train_neg_filename: The path of the training data containing negative tweets
    :param test_data_filename: The path of the test data containing unlabelled tweets
    :return: The 700 dimensions representation of the positive training set
    :return: The 700 dimensions representation of the negative training set
    :return: The 700 dimensions representation of the test data
    """
    train_pos = open(train_pos_filename, "r", encoding='utf-8')
    train_pos = [line for line in train_pos.readlines()]
    train_neg = open(train_neg_filename, 'r', encoding='utf-8')
    train_neg = [line for line in train_neg.readlines()]
    test_data = open(test_data_filename, 'r', encoding='utf-8')
    id_ = [line[:line.find(",")] for line in test_data.readlines()]
    test_data = open("test_data.txt", "r", encoding='utf-8')
    test = [line[line.find(",")+1:].rstrip('\n') for line in test_data.readlines()]
    neg_embeddings = get_sentence_embeddings(train_neg)
    pos_embeddings = get_sentence_embeddings(train_pos)
    test_embeddings = get_sentence_embeddings(test)
    return neg_embeddings, pos_embeddings, test_embeddings


def word_embedding(embedding_filename, vocab_filename):
    """
    Extract the embedding of the given vocabulary from an external embedding dataset
    :param embedding_filename: The path of the external embedding dataset
    :param vocab_filename:  The path of the dictionary we would like to get the embedding for
    :return: The embedding of all the word of the vocabulary that were present in the external dataset
    """
    df = open(embedding_filename, "r", encoding='utf-8')
    df = [line.split(" ") for line in df.readlines()]
    df = df[1:]
    # vocab = open("vocab_cut.txt", "r", encoding='utf-8')
    vocab = open(vocab_filename, "r", encoding='utf-8')
    vocab = [line.split("\n")[0] for line in vocab.readlines()]
    vocab[:] = [value for value in vocab if len(value) != 1 and ("<" not in value or ">" not in value) and not value.isdigit()]
    w_emb = []
    for word in df:
        if word[0] in vocab:
            w_emb.append(word)
    return w_emb

def sentence_avg_representation(train_pos, train_neg, test_data, embedding):
    """
    Embedding of all the data using the average of the embedding of all word in the tweet
    :param train_pos: The training data containing positive tweets
    :param train_neg: The training data containing negative tweets
    :param test_data: The test data containing unlabeled tweets
    :param embedding: The embedded vocabulary
    :return:  The tweet embedding of all tweet passed in parameters using the word average representation
    """
    global word_vector
    word_vector = embedding
    pool = Pool()
    train_neg_mean = pool.map(avg_words, train_neg)
    train_pos_mean = pool.map(avg_words, train_pos)
    test_mean = pool.map(avg_words, test_data)
    neg_label = np.ones(len(train_neg))*-1
    pos_label = np.ones(len(train_pos))*1
    train_data_mean = np.vstack((train_neg_mean, train_pos_mean))
    train_labels = np.hstack((neg_label, pos_label))
    return train_data_mean, train_labels, test_mean
