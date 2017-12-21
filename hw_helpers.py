import csv
import matplotlib.pyplot as plt
import numpy as np

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def plot_train_test(train_errors, test_errors, alphas, nb_neurones):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.figure()
    plt.semilogx(alphas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(alphas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.yticks(np.arange(0.855, 0.885, 0.01))
    plt.title("MLP Classifier on sentence embedding with nb_neurones=" + str(nb_neurones))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("validation"+str(nb_neurones)+".jpg")
