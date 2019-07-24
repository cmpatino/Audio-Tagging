from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import data_utils as du
import models

DATA_PATH = '../data/input/'


def train(model_name, data_config, model_config):

    train_df = du.create_train_df(DATA_PATH)
    print(train_df.label_idx.head())

    try:
        model = getattr(models, f'get_{model_name}')(model_config)
    except AttributeError:
        raise ValueError(f'There is no creation function for {model_name}')

    skf = StratifiedKFold(model_config.n_folds)
    skf.get_n_splits(train_df.index)

    for train_split, val_split in skf.split(train_df.index, train_df.label_idx):

        train_set = train_df.iloc[train_split]
        val_set = train_df.iloc[val_split]

        data_config = du.DataConfig()
        train_generator = du.make_train_generator(train_set,
                                                  DATA_PATH,
                                                  data_config)
        val_generator = du.make_train_generator(val_set,
                                                DATA_PATH,
                                                data_config)

        directory = f'../models/{model_name}/'
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


model_name = 'dummy_model'
data_config = du.DataConfig()
model_config = models.ModelConfig(n_folds=2)

train(model_name, data_config, model_config)