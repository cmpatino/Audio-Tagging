from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import shutil
import numpy as np
import pandas as pd

import data_utils as du
import models

DATA_PATH = '../data/input/'


def train_with_val(model_name, data_config, model_config):

    train_df = du.create_train_df(DATA_PATH)

    try:
        model = getattr(models, f'get_{model_name}')(model_config, data_config)
    except AttributeError:
        raise ValueError(f'There is no creation function for {model_name}')

    train_set, val_set = train_test_split(train_df,
                                          test_size=0.1, random_state=42)

    data_config = du.DataConfig()
    train_generator = du.make_train_generator(train_set,
                                              DATA_PATH,
                                              data_config)
    val_generator = du.make_train_generator(val_set,
                                            DATA_PATH,
                                            data_config)

    #map3_callback = Map3Callback(train_generator, train_set.label_idx,
    #                             val_generator, val_set.label_idx,
    #                             train_df.label_idx)
    directory = f'../models/{model_name}/'
    if os.path.exists(directory):
        replace = input(f'Do you want to replace the current {directory} dir? [Y/N]')
        if replace == 'N':
            raise FileExistsError('Directory already exists')
        else: 
            shutil.rmtree(directory)
            os.mkdir(directory)
    else:
        os.mkdir(directory)
    checkpoint = ModelCheckpoint(directory + 'best.h5',
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True)
    history = model.fit_generator(train_generator,
                                  validation_data=val_generator,
                                  epochs=model_config.max_epochs,
                                  use_multiprocessing=True, workers=2,
                                  max_queue_size=20,
                                  callbacks=[checkpoint])
    #history.history['map3_val'] = map3_callback.map3_val
    #history.history['map3_train'] = map3_callback.map3_train


def train_for_submission(model_name, data_config, model_config):

    train_df = du.create_train_df(DATA_PATH)

    try:
        model = getattr(models, f'get_{model_name}')(model_config, data_config)
    except AttributeError:
        raise ValueError(f'There is no creation function for {model_name}')

    data_config = du.DataConfig()
    train_generator = du.make_train_generator(train_df,
                                              DATA_PATH,
                                              data_config)

    directory = f'../models/{model_name}/'

    history = model.fit_generator(train_generator,
                                  epochs=model_config.max_epochs,
                                  use_multiprocessing=True, workers=2,
                                  max_queue_size=20)

    model.save_weights(directory + 'best_for_submission.h5')


def make_submission(model_name):

    directory = f'../models/{model_name}/'

    data_config = du.DataConfig()
    train_for_submission(model_name, data_config, model_config)
    
    try:
        model = getattr(models, f'get_{model_name}')(model_config)
    except AttributeError:
        raise ValueError(f'There is no creation function for {model_name}')
    model.load_weights(directory + 'best_for_submission.h5')

    test_generator = du.make_pred_generator(DATA_PATH,
                                            data_config)

    predictions = model.predict_generator(test_generator, verbose=1)

    # Make a submission file
    train_df = du.create_train_df(DATA_PATH)
    test_df = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    test_df = test_df[['fname']].copy()
    labels = list(train_df.label.unique())
    top_3 = np.array(labels)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_df['label'] = predicted_labels
    test_df.to_csv(f'../submissions/submission_{model_name}.csv', index=False)


model_name = 'dummy_model'
data_config = du.DataConfig()
model_config = models.ModelConfig(max_epochs=2)

train_with_val(model_name, data_config, model_config)
#make_submission(model_name)
