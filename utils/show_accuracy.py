import numpy as np


def show_accuracy_3(y_hat, y_test, beta=1):
    total = len(y_hat)
    confusion_matrix = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            confusion_matrix[i, j] = ((y_hat == i) & (y_test == j)).sum()
    correct = (y_hat == y_test).sum()
    fs_precision_recall = np.zeros([3, 3])
    for i in range(3):
        tp = confusion_matrix[i, i]
        fp, fn = 0, 0
        for j in range(3):
            if i != j:
                fn += confusion_matrix[i, j]
                fp += confusion_matrix[j, i]
        if tp == 0:
            precision, recall = 0, 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        if precision + recall == 0:
            fs = 0
        else:
            fs = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        fs_precision_recall[i, :] = [fs, precision, recall]
    acc = correct / total
    print('Accuracy of the model on the {} test samples: {} %'
          .format(total, 100 * acc))
    print('Confusion Matrix:')
    print(confusion_matrix)
    fs = np.average(fs_precision_recall[:, 0])
    print('F1-score:\t{}'.format(fs))
    print('Precision:\t{}'.format(np.average(fs_precision_recall[:, 1])))
    print('Recall:\t{}\n'.format(np.average(fs_precision_recall[:, 2])))

    return np.array([acc, fs])