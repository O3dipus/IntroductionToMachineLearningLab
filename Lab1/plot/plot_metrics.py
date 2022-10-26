import matplotlib.pyplot as plt


# calculate recall rate based on confusion matrix
# input :
#       confusion_matrix - int[][]
#       figname : figure name - string
# return :
#       None
def draw_confusion_matrix(confusion_matrix, figname):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    plt.title(figname, fontsize=20)
