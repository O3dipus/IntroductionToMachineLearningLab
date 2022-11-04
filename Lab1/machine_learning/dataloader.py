import numpy as np


# load dataset
def load_dataset(path):
    dataset = np.loadtxt(path)
    dataset = dataset[np.random.permutation(dataset.shape[0])]
    return dataset


# get cross validation  dataset
# input :
#       dataset : original dataset - float[][]
#       k : cross validation k - int
#       i : iteration number - int
#       validation_alpha : the ratio of validation dataset with respect to test dataset : float
# return :
#       train_dataset - float[][]
#       test_dataset - float[][]
#       validation_dataset - float[][]
def get_cross_validation_dataset(dataset, m, n, i, j):
    test_dataset, train_dataset = get_division_n(dataset, m, i)
    validation_dataset, train_dataset = get_division_n(train_dataset, n, j)
    return train_dataset, test_dataset, validation_dataset


def get_division_n(dataset, div, n):
    dataset_len = dataset.shape[0] / div
    div1 = dataset[int(n * dataset_len):int((n + 1) * dataset_len)]
    div2 = np.concatenate((dataset[:int(n * dataset_len)],
                           dataset[int((n + 1) * dataset_len):]), axis=0)
    return div1, div2
