from hw_helpers import plot_train_test
from training import validation_s2v
import numpy as np


train_data = np.vstack((np.loadtxt("pos_embeddings.csv", delimiter=","), np.loadtxt("neg_embeddings.csv", delimiter=",")))
pos_label = np.ones(1250000)
neg_label = np.ones(1250000)*-1
train_labels = np.hstack((pos_label, neg_label))
alphas = np.logspace(-3, -1, 5)
depth = 3
neurones = range(30, 90, 20)
parameters, train_scores, validation_scores = validation_s2v(train_data=train_data, train_labels=train_labels, alphas=alphas, depth=depth, neurones=neurones)
indices = [len(neurones)*x for x in range(len(alphas))]
for i in range(len(neurones)):
    tmp_tr_scores = train_scores[np.array(indices)+i]
    tmp_val_scores = validation_scores[np.array(indices)+i]
    plot_train_test(tmp_tr_scores, tmp_val_scores, alphas, neurones[i])
