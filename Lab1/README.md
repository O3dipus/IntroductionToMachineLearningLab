# IML Lab 1 - Decision Tree

### File Structure

```
Lab1
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
|-- test
    |-- test_data_file.txt
|-- model
    |-- <random uuid>
    	|-- model_name.txt
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
  --mode MODE           1 cross validation 2 single train 3 test 4 visualize
  --k_fold K_FOLD       the number of k in k-fold cross validation
  --draw_confusion DRAW_CONFUSION
                        1 draw confusion matrix and store in folder named fig
  --show_training_process SHOW_TRAINING_PROCESS
                        1 show accuracy of every training process 2 do not
                        show
  --train_clean TRAIN_CLEAN
                        1 training on clean dataset 2 noisy dataset
```

to be more specific , if you want to train on the clean dataset and use 10-fold cross validation, the command is shown below:

```bash
python main.py  --k_fold 10 --train_clean 1 --show_training_process 1
```

`--show_training_process` means to show accuracy of every training process.

### Model

During every training models are saved in the model folder with a random uuid name.

```
|-- model
    |-- <random uuid>
    	|-- model_name.txt
```

the format of the model name is related to local time.



### Test With Your Own Dataset

With `--mode 3` you can load a trained model and test dataset in the `test` folder. There must be exactly one test data file in the folder or it will end with no result.



### Record Confusion Matrix

Using argument `--draw_confusion` it is not difficult to record confusion matrix generated during the training process and figures will be stored in the `fig` folder.
