# IML Lab 1 - Decision Tree

### File Structure

```
Lab1
|-- fig
|-- machine_learning
    |-- __init__.py
    |-- decision_tree.py
    |-- decision_tree_node.py
    |-- dataloader.py
    |-- metrics.py
|-- plot
    |-- __init__.py
    |-- plot_metrics.py
|-- wifi_db
    |-- clean_dataset.py
    |-- noisy_dataset.py
main.py
```



### Developement Environment

**OS :** Windows 10

**IDE :** Pycharm Professional 

**Python:** 3.9

**Requirements :** numpy, matplotlib



### How to run

**step 1 :** install essential requirements mentioned above

**step 2 :** run **main.py** in the root directory



### Run With Arguments

By running command:

```python
python main.py -h 
```

 We can see all the arguments related to decision tree training:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --k_fold K_FOLD       the number of k in k-fold cross validation
  --draw_confusion DRAW_CONFUSION
                        draw confusion matrix and store in folder named fig
  --show_training_process SHOW_TRAINING_PROCESS
                        show accuracy of every training process
  --train_clean TRAIN_CLEAN
                        training on clean dataset or noisy dataset
  --validation_ratio VALIDATION_RATIO
                        the ratio of validation dataset with respect to test dataset
```

to be more specific , if you want to train on the clean dataset and use 10-fold cross validation, the command is shown below:

```bash
python main.py  --k_fold 10 --train_clean 1 --show_training_process 1 --validation_ratio 0.5
```

`--show_training_process` means to show accuracy of every training process and `--validation_ratio` is related to how you split the validation dataset which is used to do the pruning which is also recommended to be set 0.6.



### Record Confusion Matrix

Using argument `--draw_confusion` it is not difficult to record confusion matrix generated during the training process and figures will be stored in the `fig` folder.

One of the result is shown below. Every time you run the program with `--draw_confusion 1` will lead to an overwrite action to the previous figure so don't forget to store your necessary experiment data before starting the next training.

![image-20221027010236225](C:\Users\89748\AppData\Roaming\Typora\typora-user-images\image-20221027010236225.png)