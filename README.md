# G-CNN-P5
P5: Group equivariant deep learning
All the files used in the project can be found here.

log.txt -> Logs for our git
shortlog.txt -> short log for our git
confusion_results.csv -> Results for our cross validation
requirements.txt -> Python library requirements
results.csv -> results of our training
results_grid_search.csv -> results of our grid search
training_log.txt -> the log from training our models
training_log_grid_search.txt -> the training log from our grid search
/visualizations -> some visualizations made for the report
/data -> where data is located. we only use /raw in here
/models -> the trained models are located here
/src -> where our code is
  GCNN-Non-optimizedTrained.ipynb -> Old theory correct GCNN but slow in computation
  MODELS.py -> Where our GCNN and CNN models are
  VisulisationsGCNN.ipynb -> Visualizations for report
  data.py -> where we import datasets
  data_augmentation.py -> where we data-augment for the CNN-augmented
  final_tests.py -> This is where we do cross validation
  main - gridsearch.py -> This code was used to conduct the grid-search (basically just an earlier version of main.py
  main.py -> This is where it all comes together for the training part
  save_data.py -> For saving the data from training
  steerable.py -> Steerable model in here
  visualize.ipynb -> A lot of final visualizations for our results chapter
  /subfiles
    groups.py -> Where we coded our groups Cyclic and Dihedral
    kernels.py -> Code for our kernels in gcnn
    layers.py -> Code for our layers in gcnn
    motivationalExample.ipynb -> notebook for motivational example
    utils.py -> interpolation for our gcnn's
