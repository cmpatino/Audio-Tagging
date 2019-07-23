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
|   +-- model_1
|   +-- model_2
|    +-- model_i
+-- notebooks
+-- scripts
+-- submissions
+-- visualizations
```

### Train a model

Inside models.py there is a class for specifying hyperparameters for a model. Also, the script contains functions that return a compiled model. Below is an example for training a model with an architecture defined in a function called `get_sample_model()`.

```python
DATA_PATH = '../data/input/'

data_config = DataConfig()

train_generator = make_train_generator(DATA_PATH, data_config)

model_config = ModelConfig()
model = get_sample_model(model_config)

history = model.fit_generator(train_generator,
                              epochs=model_config.max_epochs,
                              use_multiprocessing=True, workers=2,
                              max_queue_size=20)
```
