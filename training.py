from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import tensorflow as tf

def mlpc_model_for_s2v(train_data, train_labels, nb_neur, alpha, depth, save=False):
    hidden_layer_size = tuple(np.repeat(nb_neur, depth))
    mlpc = MLPClassifier(hidden_layer_sizes=hidden_layer_size, alpha=alpha, activation='tanh')
    mlpc.fit(train_data, train_labels)
    if save:
        joblib.dump(mlpc, "Optimal_mlpc.pkl")
    return mlpc


def rnn_model_for_w2v(batchSize,numClasses,max_length,emb_dimension,lstmUnits, embedding_matrix):
    #Create the model
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, max_length])


    data = tf.Variable(tf.zeros([batchSize, max_length, emb_dimension]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(embedding_matrix, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    return (accuracy,optimizer,prediction,input_data,labels)


def mlpc_model_for_w2v(train_data, train_labels, test_data):
    mlpc = MLPClassifier()
    mlpc.fit(train_data, train_labels)
    prediction = mlpc.predict(test_data)
    return prediction


def validation_s2v(train_data, train_labels, alphas, neurones, depth):
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
