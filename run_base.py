from embeddings import word_embedding, sentence_avg_representation
from training import mlpc_model_for_w2v
from hw_helpers import create_csv_submission

def run():
    embedding = word_embedding("embed_tweets_en_200M_200D/embedding_file.txt", "vocab_cut.txt")
    train_pos = open("train_pos_full.txt", "r")
    train_pos = [line.split(" ") for line in train_pos.readlines()]
    train_neg = open("train_neg_full.txt", "r")
    train_neg = [line.split(" ") for line in train_neg.readlines()]
    test_data = open("test_data.txt", "r", encoding='utf-8')
    id_ = [line[:line.find(",")] for line in test_data.readlines()]
    test_data = open("test_data.txt", "r", encoding='utf-8')
    test = [line[line.find(",")+1:].strip().split(" ") for line in test_data.readlines()]
    train_data, train_labels, test_mean = sentence_avg_representation(train_pos, train_neg, test, embedding)
    prediction = mlpc_model_for_w2v(train_data, train_labels, test_mean)
    create_csv_submission(id_, prediction, "baseline_submission.csv")
    return 0

if __name__ == "__main__":
    run()
