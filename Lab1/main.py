import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot.visualization import visualize
from machine_learning.decision_tree import DecisionTree
from machine_learning.dataloader import load_dataset, get_cross_validation_dataset
from machine_learning.metrics import recall_rate, precision_rate, F1_rate
from plot.plot_metrics import draw_confusion_matrix

clean_path = 'wifi_db/clean_dataset.txt'
noisy_path = 'wifi_db/noisy_dataset.txt'


def k_fold_cross_validation(k_fold, clean=True,
                            draw_confusion=False,
                            show_training_process=False):
    if clean:
        train_label = 'clean'
        path = clean_path
    else:
        train_label = 'noisy'
        path = noisy_path
    dataset = load_dataset(path)

    outer_k = k_fold
    inner_k = k_fold - 1

    average_acc_before_pruning_list = []
    average_acc_after_pruning_list = []

    categories = np.unique(dataset[:, dataset[0].shape[0] - 1])
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

            acc_before_pruning = dt.evaluate(test_dataset)
            aver_acc_before_pruning += acc_before_pruning
            aver_recall_before_pruning += recall_rate(dt.metrics)
            aver_precision_before_pruning += precision_rate(dt.metrics)
            aver_F1_before_pruning += F1_rate(dt.metrics)
            aver_confusion_mat_before_pruning += dt.metrics
            aver_height_before_pruning += dt.get_height()

            # pruning
            dt.pruning(dt.root, validation_dataset=validation_dataset)
            acc_after_pruning = dt.evaluate(test_dataset)
            aver_acc_after_pruning += acc_after_pruning
            aver_recall_after_pruning += recall_rate(dt.metrics)
            aver_precision_after_pruning += precision_rate(dt.metrics)
            aver_F1_after_pruning += F1_rate(dt.metrics)
            aver_confusion_mat_after_pruning += dt.metrics
            aver_height_after_pruning += dt.get_height()

        average_acc_before_pruning_list.append(aver_acc_before_pruning / inner_k)
        average_acc_after_pruning_list.append(aver_acc_after_pruning / inner_k)

        if show_training_process:
            print()
            print('Iteration %d' % i)
            print('Acc Before Pruning : {}'.format(aver_acc_before_pruning / inner_k))
            print('Acc After Pruning : {}'.format(aver_acc_after_pruning / inner_k))

    aver_confusion_mat_before_pruning = np.round((aver_confusion_mat_before_pruning / (outer_k * inner_k)), 3)
    aver_confusion_mat_after_pruning = np.round((aver_confusion_mat_after_pruning / (outer_k * inner_k)), 3)

    if draw_confusion:
        draw_confusion_matrix(aver_confusion_mat_before_pruning, figname='Average Confusion Matrix Before Pruning')
        plt.savefig('./fig/confusion_matrix/%s/%s_before_pruning_average' % (train_label, train_label))
        draw_confusion_matrix(aver_confusion_mat_after_pruning, figname='Average Confusion Matrix After Pruning')
        plt.savefig('./fig/confusion_matrix/%s/%s_after_pruning_average' % (train_label, train_label))

    print()
    print('Final Result for {}: '.format(train_label))
    print()
    print('Average Acc Before Pruning: {}'.format(np.mean(average_acc_before_pruning_list)))
    print('Average Acc After Pruning: {}'.format(np.mean(average_acc_after_pruning_list)))
    print('Average Depth Before Pruning: {}'.format(aver_height_before_pruning / (outer_k * inner_k)))
    print('Average Depth After Pruning: {}'.format(aver_height_after_pruning / (outer_k * inner_k)))

    print()
    print("Average Precision Rate Before Pruning : %s" % (precision_rate(aver_confusion_mat_before_pruning)))
    print("Average Recall Rate Before Pruning : %s" % (recall_rate(aver_confusion_mat_before_pruning)))
    print("Average F1 Rate Before Pruning : %s" % (F1_rate(aver_confusion_mat_before_pruning)))

    print()
    print("Average Precision Rate After Pruning : %s" % (precision_rate(aver_confusion_mat_after_pruning)))
    print("Average Recall Rate After Pruning : %s" % (recall_rate(aver_confusion_mat_after_pruning)))
    print("Average F1 Rate After Pruning : %s" % (F1_rate(aver_confusion_mat_after_pruning)))

    return aver_acc_before_pruning / k_fold, aver_acc_after_pruning / k_fold


def train(clean=True):
    if clean:
        path = clean_path
        print('Clean dataset single train:')
    else:
        path = noisy_path
        print('Noisy dataset single train:')
    dataset = load_dataset(path)
    train_dataset, test_dataset, validation_dataset = get_cross_validation_dataset(dataset, 10, 9, 0, 0)
    dt = DecisionTree(train_dataset, test_dataset)
    dt.fit()
    print("Accuracy Before Pruning : %.3f" % dt.evaluate(test_dataset))
    dt.pruning(dt.root, validation_dataset)
    print("Accuracy After Pruning : %.3f" % dt.evaluate(test_dataset))


def visualize_data():
    dataset = load_dataset(clean_path)
    dt = DecisionTree(dataset, None)
    dt.fit()
    visualize(dt, "fig/clean_tree_visualization.png")


def main():
    np.random.seed(42)
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=int, help='1 cross validation 2 single train 3 visualize a tree trained on '
                                                 'entire clean dataset', default=1)
    parser.add_argument('--k_fold', type=int, help='the number of k in k-fold cross validation', default=10)
    parser.add_argument('--draw_confusion', type=int, help='1 draw confusion matrix figure and save it',
                        default=1)
    parser.add_argument('--show_training_process', type=int,
                        help='1 print accuracy for all k iterations',
                        default=1)
    parser.add_argument('--dataset', type=int, help='1 clean dataset 2 noisy dataset 3 both', default=3)

    args = parser.parse_args()

    if args.mode == 1:
        if args.dataset == 3:
            k_fold_cross_validation(args.k_fold,
                                    draw_confusion=args.draw_confusion == 1,
                                    show_training_process=args.show_training_process == 1)
            k_fold_cross_validation(args.k_fold,
                                    draw_confusion=args.draw_confusion == 1,
                                    show_training_process=args.show_training_process == 1,
                                    clean=False)
        else:
            k_fold_cross_validation(args.k_fold,
                                    draw_confusion=args.draw_confusion,
                                    show_training_process=args.show_training_process == 1,
                                    clean=args.dataset == 1)

    elif args.mode == 2:
        if args.dataset == 3:
            train()
            train(False)
        else:
            train(clean=(args.dataset == 1))

    elif args.mode == 3:
        visualize_data()


if __name__ == "__main__":
    main()
