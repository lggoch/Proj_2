from sent2vec import get_sentence_embeddings
from sklearn.externals import joblib
import numpy as np
from hw_helpers import create_csv_submission


def run():
    test_data = open("test_data.txt", "r", encoding='utf-8')
    test = [line[line.find(",")+1:].rstrip('\n') for line in test_data.readlines()]
    test_data = open("test_data.txt", "r", encoding='utf-8')
    id_ = [line[:line.find(",")] for line in test_data.readlines()]
    test_embedding = get_sentence_embeddings(test)
    clf = joblib.load("Optimal_mlpc.pkl")
    prediction = clf.predict(test_embedding)
    id_ = np.array(id_).astype("int")
    create_csv_submission(id_, prediction, "kaggle_submission.csv")
    return 0


if __name__ == "__main__":
    run()

