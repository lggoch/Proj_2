from embeddings import sentence_embedding
import numpy as np
from training import mlpc_model_for_s2v
from hw_helpers import create_csv_submission
#load the model

#load the embedding or the twitter bigram

#embedd the test data

#make the prediction

#return the csv submission file

def run():
    neg_embeddings, pos_embeddings, test_embeddings = sentence_embedding("train_pos_full.txt", "train_neg_full.txt", "test_data.txt")
    train_data = np.vstack((neg_embeddings, pos_embeddings))
    pos_labels = np.ones(len(pos_embeddings))
    neg_labels = np.ones(len(neg_embeddings))*-1
    train_labels = np.hstack((pos_labels, neg_labels))
    clf = mlpc_model_for_s2v(train_data=train_data, train_labels=train_labels, nb_neur=50, alpha=0.001, depth=5, save=True)
    prediction = clf.predict(test_embeddings)
    test_data = open("test_data.txt", "r", encoding='utf-8')
    id_ = [line[:line.find(",")] for line in test_data.readlines()]
    create_csv_submission(id_)
    id_ = np.array(id_).astype("int")
    create_csv_submission(id_, prediction, "s2v_a0001_d5_n50_submission.csv")
    return 0

if __name__ == "__main__":
    run()
