from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np


def mlpc_model_for_s2v(train_data, train_labels, nb_neur, alpha, depth, save=False):
    """
    Create the Multi-Layer Perceptron classifier for the sent2vec embedding data
    :param train_data: The training data (positive and negative) embedded via the sent2vec algorithm
    :param train_labels: The labels of the training data (-1 for negative, 1 for positive)
    :param nb_neur: The number of neurones per hidden layer
    :param alpha: The alpha regularization parameter
    :param depth: The depth of the neural network
    :param save: Whether you would like to same the model or not
    :return: The trained model ready to make prediction
    """
    hidden_layer_size = tuple(np.repeat(nb_neur, depth))
    mlpc = MLPClassifier(hidden_layer_sizes=hidden_layer_size, alpha=alpha, activation='tanh')
    mlpc.fit(train_data, train_labels)
    if save:
        joblib.dump(mlpc, "Optimal_mlpc.pkl")
    return mlpc


def rnn_model_for_w2v():
    return 0


def mlpc_model_for_w2v(train_data, train_labels, test_data):
    """
    Create the Multi-Layer Perceptron classifier for the average with word2vec embedding data
    :param train_data: The embedded training data via averaging all the word embeddings of a tweet
    :param train_labels: The labels of the training data (-1 for negative, 1 for positive)
    :param test_data: The embedded test data via averaging all the word embeddings of a tweet
    :return:The prediction labels for the test data
    """
    mlpc = MLPClassifier()
    mlpc.fit(train_data, train_labels)
    prediction = mlpc.predict(test_data)
    return prediction


def validation_s2v(train_data, train_labels, alphas, neurones, depth):
    """
    Separate the train_data into a train and a validation set. The validation set is used to choose hyper parameters.
    :param train_data:The embedded training data via sent2vec
    :param train_labels: The labels of the training data (-1 for negative, 1 for positive)
    :param alphas: The different alphas to test
    :param neurones: The different K to test, where K is the number of neurones per hidden layer
    :param depth: The depth of the neural network
    :return: The array containing all pair of hyperparameters that were tested
    :return: The array containing the training accuracy for all pair of hyperparameters
    :return: The array containing the validation accuracy for all pair of hyperparameters
    """
    hidden_layer_size = tuple(np.repeat(neurones, depth))
    train_scores = []
    test_scores = []
    parameters = []
    for alpha in alphas:
        for neur in neurones:
            X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.1, random_state=40)
            mlpc = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_size, activation='tanh', batch_size=5000)
            mlpc.fit(X_train, y_train)
            print("Parameters:", alpha, neur)
            parameters.append((alpha, neur))
            train_score = mlpc.score(X_train, y_train)
            train_scores.append(train_score)
            print("Train_score: %0.4f" % train_score)
            test_score = mlpc.score(X_test, y_test)
            test_scores.append(test_score)
            print("Test_score: %0.4f" % test_score)
    return parameters, train_scores, test_scores
