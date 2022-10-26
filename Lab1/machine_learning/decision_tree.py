import numpy as np
import copy
from machine_learning.decision_tree_node import DecisionTreeNode
from machine_learning.metrics import calculate_H


class DecisionTree:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.label_index = self.train_dataset.shape[1] - 1
        self.labels = np.unique(self.train_dataset[:, self.label_index])
        self.max_depth = 0
        # TP, FP, TN, FN
        self.metrics = np.zeros([self.labels.shape[0], self.labels.shape[0]])
        self.root = None

    def fit(self):
        self.root = self.decision_tree_learning(self.train_dataset, 1)
        return self.root

    # decision tree training
    # input :
    #       train_dataset - float[][]
    #       depth : tree depth - int
    # return :
    #       node : current node - DecisionTreeNode
    def decision_tree_learning(self, train_dataset, depth):
        self.max_depth = max(self.max_depth, depth)
        label_ind = train_dataset.shape[1] - 1
        if all(train_dataset[:, label_ind] == train_dataset[0, label_ind]):
            node = DecisionTreeNode.label_node(label=train_dataset[0, label_ind],
                                               sample_size=train_dataset.shape[0])
            node.set_depth(depth)
            return node
        else:
            node, left_ds, right_ds = self.find_split(train_dataset)

            if node.is_leaf:
                return node

            left_child = self.decision_tree_learning(left_ds, depth + 1)
            right_child = self.decision_tree_learning(right_ds, depth + 1)
            node.set_left_child(left_child)
            node.set_right_child(right_child)
            node.set_depth(depth)

            if self.root is None:
                self.root = node

            return node

    # decision tree training
    # input :
    #       test_dataset - float[][]
    # return :
    #       accuracy - float
    def validate(self, test_dataset):
        # print("Test Dataset Size : (%d, %d)" % (test_dataset.shape[0], test_dataset.shape[1]))
        accuracy = 0
        self.metrics = np.zeros([self.labels.shape[0], self.labels.shape[0]])
        for s in test_dataset:
            res = self.root.judge(s)
            label = s[self.label_index]
            if res == label:
                self.metrics[int(label) - 1, int(label) - 1] += 1
                accuracy = accuracy + 1
            else:
                self.metrics[int(label) - 1, int(res) - 1] += 1
        accuracy = accuracy / test_dataset.shape[0]

        return accuracy

    # find split point in dataset
    # input :
    #       dataset - float[][]
    # return :
    #       node : split node - DecisionTreeNode
    #       left_ds : left dataset - float[][]
    #       right_ds : right dataset - float[][]
    def find_split(self, dataset):
        ds = copy.deepcopy(dataset)
        H_all = calculate_H(ds)
        S_total = ds.shape[0]

        max_gain, max_gain_dim, max_gain_index, max_gain_value = 0, -1, -1, -1
        for i in range(dataset.shape[1] - 1):
            temp = ds[ds[:, i].argsort()]
            for ind in range(1, temp.shape[0]):
                H_left = calculate_H(temp[:ind, :])
                H_right = calculate_H(temp[ind:, :])

                remainder = (ind / S_total) * H_left + ((S_total - ind) / S_total) * H_right
                gain = H_all - remainder

                if gain > max_gain:
                    max_gain, max_gain_dim, max_gain_index, max_gain_value = gain, i, ind, temp[ind, i]

        ds = ds[ds[:, max_gain_dim].argsort()]
        left_ds = ds[:max_gain_index, :]
        right_ds = ds[max_gain_index:, :]
        label_count = np.array([(dataset[dataset[:, self.label_index] == label]).shape[0] for label in self.labels])

        node = DecisionTreeNode.decision_node(dim=max_gain_dim,
                                              threshold=max_gain_value,
                                              label_count=label_count)
        return node, left_ds, right_ds

    # pruning in decision tree
    # input :
    #       node : current node - DecisionTreeNode
    #       validation_dataset - float[][]
    #       random_branching : left_most_indexing or random indexing to visit the tree - bool
    # return :
    #       None
    def pruning(self, node, validation_dataset, random_branching=False):
        if node is None:
            return None

        if random_branching:
            left_first = np.random.rand() > 0.5
            if left_first:
                self.pruning(node.left, validation_dataset)
                self.pruning(node.right, validation_dataset)
            else:
                self.pruning(node.right, validation_dataset)
                self.pruning(node.left, validation_dataset)
        else:
            self.pruning(node.left, validation_dataset)
            self.pruning(node.right, validation_dataset)

        if node.left is None or node.right is None:
            return
        elif not node.left.is_leaf or not node.right.is_leaf:
            return
        else:
            # get current accuracy of the validation dataset
            acc = self.validate(validation_dataset)

            node.is_leaf = True
            # apply left node value
            node.label = node.left.label
            acc_left = self.validate(validation_dataset)
            # apply right node value
            node.label = node.right.label
            acc_right = self.validate(validation_dataset)
            if acc > max(acc_right, acc_left):
                node.is_leaf = False
                node.label = None
            else:
                if acc_left >= acc:
                    acc = acc_left
                    node.label = node.left.label
                if acc_right >= acc:
                    node.label = node.right.label
