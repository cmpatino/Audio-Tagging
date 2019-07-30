# Audio-Tagging

## Project Overview

This repository contains the solution for the [Freesound General-Purpose Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging/overview/) competition on Kaggle.

The objective of the challenge is to predict the 3 most likely labels for a a given audio. The task is evaluated using the MAP@3 metric.

## How to run

The directory structure of the repository is shown in the diagram.

```
+-- data
|   +-- input
+-- models
|   +-- model_name_1
|   +-- model_name_2
|   +-- model_name_3
+-- notebooks
+-- scripts
+-- submissions
+-- visualizations
```

There are two modes to run training: 
+ Training with validation set using the function `train_with_val()`
+ Training for best submission  using the function `make_submission()`

The mode that creates a validation set is ideal for testing the performance of the model using the validation set. Measuring the performance metrics on the validation set helps avoid problems such as overfitting or underfitting.

The mode with trainig for best submission does not split the training data into train and test sets. Instead, the entirety of the training data is used to train the model. This mode is ideal for maximizing the peformance of the model on the Kaggle score when the best hyperparameters are already known.