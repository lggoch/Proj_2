"""
Code reference: https://github.com/adeshpande3/LSTM-Sentiment-Analysis
Code reference: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

"""


from helper_rnn import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from training import rnn_model_for_w2v
import tensorflow as tf

def run():
    """
    Run the RNN model using and output a CSV file for kaggle Submission.

    """
    batchSize = 1000
    lstmUnits = 80
    iterations = 10000
    output_filename = r"lstm_unit" + str(lstmUnits) + "_bs_" + str(batchSize) +"_i_"+str(iterations) +".csv"
    pos_filename = "train_pos_full.txt"
    neg_filename = "train_neg_full.txt"
    test_filename = "test_data.txt"
    numClasses = 2
    emb_dimension = 200


    (id, test_line) = generate_test(test_filename) #separate id and test sentences

    nbrTest = len(id)

    embedding = word_embedding("embed_tweets_en_200M_200D/embedding_file.txt", "vocab_cut.txt")

    train_pos = sep_line(pos_filename)
    train_neg = sep_line(neg_filename)


    nbr_tweet = len(train_pos) # Number of tweet in each connotation


    (sentences, labels_) = extract_data(train_pos,train_neg,0, nbr_tweet)

    # Map an index to words
    t = Tokenizer()
    t.fit_on_texts(sentences+test_line)
    (enc_docs_train, max_length_train) = generate_input(sentences, t)
    (enc_docs_test, max_length_test) = generate_input(test_line, t)


    max_length = max(max_length_train,max_length_test)
    padded_docs_train = pad_sequences(enc_docs_train, maxlen=max_length, padding='post')
    padded_docs_test = pad_sequences(enc_docs_test, maxlen=max_length, padding='post')
    vocab_size = len(t.word_index) + 1

    # Trasnform a list of embedding into a dictionnary where the keys are words and the valus are the embedded vector.
    embedding_array = np.asarray(embedding)
    keys = embedding_array[:,0]
    values = embedding_array[:,1:]
    zip = list(keys,values)
    embeddings_index = dict(zip)



    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, emb_dimension), dtype=np.float32)
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector



    (accuracy,optimizer,prediction, input_data,labels) = rnn_model_for_w2v(batchSize, numClasses, max_length, emb_dimension, lstmUnits, embedding_matrix)

    print(r"Start Training")

    tf.summary.scalar('Accuracy', accuracy)

    sess = tf.InteractiveSession()




    sess.run(tf.global_variables_initializer())


    # Train the model
    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch(batchSize, max_length, padded_docs_train, nbr_tweet);
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if (i % int(iterations / 10) == 0):
            summary = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})

            print("%d th iteration with accuracy %s" % (i, summary))


    # Prediction
    res = np.zeros((nbrTest, 1))

    iteration_list = np.ndarray.tolist(np.arange(batchSize, nbrTest, batchSize))
    prev = 0
    for i in iteration_list:
        print(i)
        classification = sess.run(tf.argmax(prediction, 1),
                                  feed_dict={input_data: padded_docs_test[prev:i]})  # position of the label
        classification[classification == 1] = -1
        classification[classification == 0] = 1
        classification = classification.reshape(batchSize,1)
        res[prev:i]=classification
        prev = i

    create_csv_submission(id, res, output_filename)


    return 0

if __name__ == "__main__":

    run()
