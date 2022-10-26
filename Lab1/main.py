import argparse

import matplotlib.pyplot as plt
import numpy as np

from machine_learning.decision_tree import DecisionTree
from machine_learning.dataloader import load_dataset, get_cross_validation_dataset
from machine_learning.metrics import recall_rate, precision_rate, F1_rate
from plot.plot_metrics import draw_confusion_matrix


def k_fold_cross_validation(k_fold, clean=True,
                            draw_confusion=False,
                            show_training_process=False,
                            validation_alpha=0.6):
    if clean:
        print(clean)
        train_label = 'clean'
        path = 'wifi_db/clean_dataset.txt'
    else:
        train_label = 'noisy'
        path = 'wifi_db/noisy_dataset.txt'
    dataset = load_dataset(path)

    aver_acc_before_pruning = 0
    aver_recall_before_pruning = np.zeros(4)
    aver_precision_before_pruning = np.zeros(4)
    aver_F1_before_pruning = np.zeros(4)

    aver_acc_after_pruning = 0
    aver_recall_after_pruning = np.zeros(4)
    aver_precision_after_pruning = np.zeros(4)
    aver_F1_after_pruning = np.zeros(4)

    for i in range(k_fold):
        train_dataset, test_dataset, validation_dataset = get_cross_validation_dataset(dataset, k_fold, i,
                                                                                       validation_alpha=validation_alpha)

        dt = DecisionTree(train_dataset, test_dataset)
        dt.fit()

        acc_before_pruning = dt.validate(test_dataset)
        aver_acc_before_pruning += acc_before_pruning
        aver_recall_before_pruning += recall_rate(dt.metrics)
        aver_precision_before_pruning += precision_rate(dt.metrics)
        aver_F1_before_pruning += F1_rate(dt.metrics)

        if draw_confusion:
            draw_confusion_matrix(dt.metrics, figname='%s Dataset(Before Pruning)' % train_label)
            plt.savefig('./fig/%s/%s_before_pruning_iter_%d' % (train_label, train_label, i))

        # pruning
        dt.pruning(dt.root, validation_dataset=validation_dataset)
        acc_after_pruning = dt.validate(test_dataset)
        aver_acc_after_pruning += acc_after_pruning
        aver_recall_after_pruning += recall_rate(dt.metrics)
        aver_precision_after_pruning += precision_rate(dt.metrics)
        aver_F1_after_pruning += F1_rate(dt.metrics)

        if show_training_process:
            if i == 0:
                print("Train Dataset Size : (%d, %d)" % (train_dataset.shape[0], train_dataset.shape[1]))
                print("Test Dataset Size : (%d, %d)" % (test_dataset.shape[0], test_dataset.shape[1]))
                print("Validate Dataset Size : (%d, %d)" % (validation_dataset.shape[0], validation_dataset.shape[1]))

            print()
            print("Iter %d" % i)
            print("Acc Before Pruning : %.3f" % acc_before_pruning)
            print("Acc After Pruning : %.3f" % acc_after_pruning)

        if draw_confusion:
            draw_confusion_matrix(dt.metrics, figname='%s Dataset(After Pruning)' % train_label)
            plt.savefig('./fig/%s/%s_after_pruning_iter_%d' % (train_label, train_label, i))

    print()

    print("Average Accuracy Before Pruning : %.3f" % (aver_acc_before_pruning / k_fold))
    print("Average Precision Rate Before Pruning : %s" % (aver_precision_before_pruning / k_fold))
    print("Average Recall Rate Before Pruning : %s" % (aver_recall_before_pruning / k_fold))
    print("Average F1 Rate Before Pruning : %s" % (aver_F1_before_pruning / k_fold))

    print()

    print("Average Accuracy After Pruning : %.3f" % (aver_acc_after_pruning / k_fold))
    print("Average Precision Rate After Pruning : %s" % (aver_precision_after_pruning / k_fold))
    print("Average Recall Rate After Pruning : %s" % (aver_recall_after_pruning / k_fold))
    print("Average F1 Rate After Pruning : %s" % (aver_F1_after_pruning / k_fold))

    return aver_acc_before_pruning / k_fold, aver_acc_after_pruning / k_fold


def clean_dataset_pruned_efficiency_validation():
    not_pruned_acc = []
    pruned_acc = []
    counter = 0
    for i in range(20):
        aver_acc, aver_pruned_acc = k_fold_cross_validation(k, clean=True)
        not_pruned_acc.append(aver_acc)
        pruned_acc.append(aver_pruned_acc)
        plt.plot(np.arange(0, i + 1), not_pruned_acc, label='not pruned')
        plt.plot(np.arange(0, i + 1), pruned_acc, label='pruned')
        plt.ylabel('Cross Validation Average Acc')
        plt.grid()
        plt.title('Pruned vs Not Pruned')
        plt.legend()
        if i == 19:
            plt.savefig('pruned_efficiency')
        if aver_acc >= aver_pruned_acc:
            counter += 1
    plt.show()

    print(counter)


def main():
    np.random.seed(1024)
    parser = argparse.ArgumentParser()

    parser.add_argument('--k_fold', type=int, help='the number of k in k-fold cross validation')
    parser.add_argument('--draw_confusion', type=bool, help='draw confusion matrix and store in folder named fig')
    parser.add_argument('--show_training_process', type=int, help='show accuracy of every training process')
    parser.add_argument('--train_clean', type=int, help='training on clean dataset or noisy dataset')
    parser.add_argument('--validation_ratio', type=float, help='the ratio of validation dataset with respect to test '
                                                               'dataset')

    args = parser.parse_args()

    print(args)

    k_fold_cross_validation(args.k_fold,
                            draw_confusion=args.draw_confusion,
                            show_training_process=args.show_training_process == 1,
                            clean=args.train_clean == 1,
                            validation_alpha=args.validation_ratio)


if __name__ == "__main__":
    main()
