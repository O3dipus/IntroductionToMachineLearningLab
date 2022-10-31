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
    # test_dataset_len = dataset.shape[0] * test
    # test_dataset = dataset[int(i * test_dataset_len):int((i + 1) * test_dataset_len)]
    # train_dataset = np.concatenate((dataset[:int(i * test_dataset_len)],
    #                                 dataset[int((i + 1) * test_dataset_len):]), axis=0)
    validation_dataset, train_dataset = get_division_n(train_dataset, n, j)
    # train_dataset_len = train_dataset.shape[0] * validate
    # validation_dataset = dataset[int(j * train_dataset_len):int((j + 1) * train_dataset_len)]
    # train_dataset = np.concatenate((train_dataset[:int(j * train_dataset_len)],
    #                                 train_dataset[int((j + 1) * train_dataset_len):]), axis=0)
    return train_dataset, test_dataset, validation_dataset


def get_division_n(dataset, div, n):
    dataset_len = dataset.shape[0] / div
    div1 = dataset[int(n * dataset_len):int((n + 1) * dataset_len)]
    div2 = np.concatenate((dataset[:int(n * dataset_len)],
                           dataset[int((n + 1) * dataset_len):]), axis=0)
    return div1, div2
