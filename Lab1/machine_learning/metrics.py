import numpy as np


# find split point in dataset
# input :
#       dataset - float[][]
# return :
#       H - entropy - float
def calculate_H(dataset):
    labels = np.unique(dataset[:, 7])
    pk = [len(dataset[dataset[:, 7] == label]) / len(dataset) for label in labels]
    H = 0
    for p in pk:
        if p == 0:
            continue
        H = H - p * np.log2(p)
    return H


# calculate recall rate based on confusion matrix
# input :
#       confusion_matrix - int[][]
# return :
#       recall - float
def recall_rate(confusion_matrix):
    recall = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
    return recall


# calculate precision rate based on confusion matrix
# input :
#       confusion_matrix - int[][]
# return :
#       precision - float
def precision_rate(confusion_matrix):
    precision = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
    return precision


def F1_rate(confusion_matrix):
    PPV = precision_rate(confusion_matrix)
    TPR = recall_rate(confusion_matrix)
    return 2 * PPV * TPR / (TPR + PPV)
