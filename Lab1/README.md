# Introduction to ML - Decision Tree Coursework

### File Structure

```
Coursework_1
|-- fig
    |-- confusion_matrix
        |-- clean
        |-- noiy
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
    |-- clean_dataset.txt
    |-- noisy_dataset.txt
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
  
  --mode MODE           1 cross validation
                        2 single train 
                        3 visualize a tree trained on entire clean dataset
                        
  --k_fold K_FOLD       the number of k of k-fold cross validation
  
  --draw_confusion DRAW_CONFUSION
                        1 draw confusion matrix image and save it

  --show_training_process SHOW_TRAINING_PROCESS
                        1 print accuracy for all k iterations
                        
  --dataset DATASET     1 clean dataset 
                        2 noisy dataset 
                        3 both
```

e.g. if you want to train on the clean dataset and use 10-fold cross validation, the command is shown below:

```bash
python main.py  --mode 1 --k_fold 10 --dataset 1
```


### Draw Confusion Matrix

Using argument `--draw_confusion` it is not difficult to record confusion matrix generated during the training process and figures will be stored in the `fig` folder.
