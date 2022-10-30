import argparse
import os
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from plot.visualization import visualize
from machine_learning.decision_tree import DecisionTree
from machine_learning.dataloader import load_dataset, get_cross_validation_dataset
from machine_learning.metrics import recall_rate, precision_rate, F1_rate
from plot.plot_metrics import draw_confusion_matrix


def k_fold_cross_validation(k_fold, clean=True,
                            draw_confusion=False,
                            show_training_process=False):
    if clean:
        train_label = 'clean'
        path = 'wifi_db/clean_dataset.txt'
    else:
        train_label = 'noisy'
        path = 'wifi_db/noisy_dataset.txt'
    dataset = load_dataset(path)

    outer_k = k_fold
    inner_k = k_fold

    average_acc_before_pruning_list = []
    average_acc_after_pruning_list = []

    model_path = uuid.uuid4()

    categories = np.unique(dataset[:, dataset[0].shape[0]-1])
    aver_confusion_mat_before_pruning = np.zeros((categories.shape[0], categories.shape[0]))
    aver_confusion_mat_after_pruning = np.zeros((categories.shape[0], categories.shape[0]))
    aver_height_before_pruning = 0
    aver_height_after_pruning = 0

    for i in range(outer_k):
        aver_acc_before_pruning = 0
        aver_recall_before_pruning = np.zeros(4)
        aver_precision_before_pruning = np.zeros(4)
        aver_F1_before_pruning = np.zeros(4)

        aver_acc_after_pruning = 0
        aver_recall_after_pruning = np.zeros(4)
        aver_precision_after_pruning = np.zeros(4)
        aver_F1_after_pruning = np.zeros(4)
        for j in range(inner_k):
            train_dataset, test_dataset, validation_dataset = get_cross_validation_dataset(dataset,
                                                                                           outer_k, inner_k,
                                                                                           i, j)
            dt = DecisionTree(train_dataset, test_dataset)
            dt.fit()

            acc_before_pruning = dt.validate(test_dataset)
            aver_acc_before_pruning += acc_before_pruning
            aver_recall_before_pruning += recall_rate(dt.metrics)
            aver_precision_before_pruning += precision_rate(dt.metrics)
            aver_F1_before_pruning += F1_rate(dt.metrics)
            aver_confusion_mat_before_pruning += dt.metrics
            aver_height_before_pruning += dt.get_height()

            if draw_confusion:
                draw_confusion_matrix(dt.metrics, figname='%s Dataset(Before Pruning)' % train_label)
                plt.savefig('./fig/confusion_matrix/%s/%s_before_pruning_iter_%d_%d' % (train_label, train_label, i, j))
                plt.close()

            visualize(dt, './fig/tree_structure/%s/%s_before_pruning_iter_%d_%d.png' % (train_label, train_label, i, j))

            # pruning
            dt.pruning(dt.root, validation_dataset=validation_dataset)
            acc_after_pruning = dt.validate(test_dataset)
            aver_acc_after_pruning += acc_after_pruning
            aver_recall_after_pruning += recall_rate(dt.metrics)
            aver_precision_after_pruning += precision_rate(dt.metrics)
            aver_F1_after_pruning += F1_rate(dt.metrics)
            aver_confusion_mat_after_pruning += dt.metrics
            aver_height_after_pruning += dt.get_height()

            dt.save_model(dirname=model_path,
                          filename="%s_%d_%d" % (datetime.now().strftime("%m_%d_%Y_%H_%M_%S"), i, j))

            if show_training_process:
                print()
                print("Iter %d %d" % (i, j))
                print("Acc Before Pruning : %.3f" % acc_before_pruning)
                print("Acc After Pruning : %.3f" % acc_after_pruning)

            if draw_confusion:
                draw_confusion_matrix(dt.metrics, figname='%s Dataset(After Pruning)' % train_label)
                plt.savefig('./fig/confusion_matrix/%s/%s_after_pruning_iter_%d' % (train_label, train_label, i))
                plt.close()

            visualize(dt, './fig/tree_structure/%s/%s_after_pruning_iter_%d_%d.png' % (train_label, train_label, i, j))

        print()
        print("Iteration %d" % i)
        print("Average Accuracy Before Pruning : %.3f" % (aver_acc_before_pruning / k_fold))
        print("Average Precision Rate Before Pruning : %s" % (aver_precision_before_pruning / k_fold))
        print("Average Recall Rate Before Pruning : %s" % (aver_recall_before_pruning / k_fold))
        print("Average F1 Rate Before Pruning : %s" % (aver_F1_before_pruning / k_fold))

        print()

        print("Average Accuracy After Pruning : %.3f" % (aver_acc_after_pruning / k_fold))
        print("Average Precision Rate After Pruning : %s" % (aver_precision_after_pruning / k_fold))
        print("Average Recall Rate After Pruning : %s" % (aver_recall_after_pruning / k_fold))
        print("Average F1 Rate After Pruning : %s" % (aver_F1_after_pruning / k_fold))
        print()

        average_acc_before_pruning_list.append(aver_acc_before_pruning / k_fold)
        average_acc_after_pruning_list.append(aver_acc_after_pruning / k_fold)

    plt.figure()
    plt.title("Cross-validation Acc")
    plt.plot(np.arange(1, k_fold + 1), average_acc_before_pruning_list, label='before pruning')
    plt.plot(np.arange(1, k_fold + 1), average_acc_after_pruning_list, label='after pruning')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("./fig/cross_validation")
    # plt.show()

    aver_confusion_mat_before_pruning = (aver_confusion_mat_before_pruning / (outer_k * inner_k))
    aver_confusion_mat_after_pruning = (aver_confusion_mat_after_pruning / (outer_k * inner_k))
    draw_confusion_matrix(aver_confusion_mat_before_pruning, figname='Average Confusion Matrix Before Pruning')
    plt.savefig('./fig/confusion_matrix/%s/%s_before_pruning_average' % (train_label, train_label))
    draw_confusion_matrix(aver_confusion_mat_after_pruning, figname='Average Confusion Matrix After Pruning')
    plt.savefig('./fig/confusion_matrix/%s/%s_after_pruning_average' % (train_label, train_label))

    print('Average Depth Before Pruning: {}'.format(aver_height_before_pruning / (outer_k * inner_k)))
    print('Average Depth After Pruning: {}'.format(aver_height_after_pruning / (outer_k * inner_k)))

    return aver_acc_before_pruning / k_fold, aver_acc_after_pruning / k_fold


def train(clean):
    if clean:
        path = 'wifi_db/clean_dataset.txt'
    else:
        path = 'wifi_db/noisy_dataset.txt'
    dataset = load_dataset(path)
    train_dataset, test_dataset, validation_dataset = get_cross_validation_dataset(dataset, 10, 10, 0, 0)
    dt = DecisionTree(train_dataset, test_dataset)
    dt.fit()
    print("Accuracy Before Pruning : %.3f" % dt.validate(test_dataset))
    dt.pruning(dt.root, validation_dataset)
    print("Accuracy After Pruning : %.3f" % dt.validate(test_dataset))


def load_model_and_test():
    model_path = './model/2b17276f-323b-4092-8c3c-54397e443092/10_27_2022_20_38_24_0_0.txt'
    model_name = '10_27_2022_20_38_24_0_0'

    paths = os.listdir('./test')
    if len(paths) > 1:
        print('Please put exactly 1 test data file in the folder')
        return
    print("Test Filename : %s" % paths[0])
    print("Model name : %s " % model_name)
    dataset = np.loadtxt('./test/%s' % paths[0])
    dt = DecisionTree(dataset, dataset)
    dt.load_model(model_path=model_path)
    print("Accuracy : {}".format(dt.validate(dataset)))


def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int, help='1 cross validation 2 single train 3 test', default=1)
    parser.add_argument('--k_fold', type=int, help='the number of k in k-fold cross validation', default=10)
    parser.add_argument('--draw_confusion', type=bool, help='draw confusion matrix and store in folder named fig',
                        default=False)
    parser.add_argument('--show_training_process', type=int, help='show accuracy of every training process',
                        default=True)
    parser.add_argument('--train_clean', type=int, help='training on clean dataset or noisy dataset', default=1)

    args = parser.parse_args()

    if args.mode == 1:
        k_fold_cross_validation(args.k_fold,
                                draw_confusion=args.draw_confusion,
                                show_training_process=args.show_training_process == 1,
                                clean=args.train_clean == 1)
    elif args.mode == 2:
        train(clean=(args.train_clean == 1))
    elif args.mode == 3:
        load_model_and_test()


if __name__ == "__main__":
    main()
