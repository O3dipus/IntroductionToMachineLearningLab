import numpy as np


# load dataset
def load_dataset(path):
    dataset = np.loadtxt(path)
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
def get_cross_validation_dataset(dataset, k, i, validation_alpha=0.6):
    dataset = dataset[np.random.permutation(dataset.shape[0])]
    test_dataset_len = dataset.shape[0] / k

    test_dataset = dataset[int(i * test_dataset_len):int((i + 1) * test_dataset_len)]

    validation_dataset = test_dataset[:int(test_dataset_len * validation_alpha)]
    test_dataset = test_dataset[int(test_dataset_len * validation_alpha):]
    train_dataset = np.concatenate((dataset[:int(i * test_dataset_len)],
                                    dataset[int((i + 1) * test_dataset_len):]), axis=0)
    return train_dataset, test_dataset, validation_dataset
