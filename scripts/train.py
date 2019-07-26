from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import shutil
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import data_utils as du
import models

DATA_PATH = '../data/input/'
PRE_DATA_TRAIN = '../data/preprocessed/train'
SPECTROGRAMS_TRAIN = '../data/spectrograms_df_train.csv'
SPECTROGRAMS_TEST = '../data/spectrograms_df_test.csv'


def train_with_val(model_name, data_config, model_config,
                   make_submission=True, augment=False):

    train_df = du.create_train_df(DATA_PATH)

    try:
        model = getattr(models, f'get_{model_name}')(model_config, data_config)
    except AttributeError:
        raise ValueError(f'There is no creation function for {model_name}')


    if not augment:

        train_set, val_set = train_test_split(train_df,
                                              test_size=0.1, random_state=42)

        train_generator = du.make_train_generator(train_set,
                                                  DATA_PATH,
                                                  data_config)
        val_generator = du.make_train_generator(val_set,
                                                DATA_PATH,
                                                data_config)

    else:
        sources_df = splitting_dataset(SPECTROGRAMS_TRAIN,
                                       valid_size=1500)
        sources_train = build_sources_from_metadata(sources_df,
                                                    PRE_DATA_TRAIN,
                                                    mode='train',
                                                    label_type='id')
        train_set = pd.DataFrame(sources_train, columns=['fname', 'label_idx'])
        train_set = train_set.set_index('fname')

        sources_val = build_sources_from_metadata(sources_df,
                                                  PRE_DATA_TRAIN,
                                                  mode='valid',
                                                  label_type='id')
        val_set = pd.DataFrame(sources_val, columns=['fname', 'label_idx'])
        val_set = val_set.set_index('fname')

        train_generator = du.make_train_generator(train_set,
                                                  DATA_PATH,
                                                  data_config, augment=True)
        val_generator = du.make_train_generator(val_set,
                                                DATA_PATH,
                                                data_config, augment=True)

    directory = f'../models/{model_name}/'
    
    if os.path.exists(directory):
        replace = input(f'Do you want to replace the current \
                        {directory} dir? [Y/N]\n')
        if replace == 'N':
            raise FileExistsError('Directory already exists')
        else:
            shutil.rmtree(directory)
            os.mkdir(directory)
    else:
        os.mkdir(directory)
    checkpoint = ModelCheckpoint(directory + 'best_train_val.h5',
                                 monitor='val_loss',
                                 verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
    history = model.fit_generator(train_generator,
                                  validation_data=val_generator,
                                  epochs=model_config.max_epochs,
                                  use_multiprocessing=False, workers=1,
                                  max_queue_size=10,
                                  callbacks=[checkpoint, early_stop])

    with open(directory + 'history_train_val.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    model.load_weights(directory + 'best_train_val.h5')

    if make_submission:
        if not augment:
            test_generator = du.make_pred_generator(DATA_PATH,
                                                    data_config)
        else:
            test_generator = du.make_pred_generator(DATA_PATH,
                                                    data_config,
                                                    augment=True)
        predictions = model.predict_generator(test_generator, verbose=1)

        # Save predictions
        np.save(directory + "raw_train_val_predictions.npy", predictions)

        # Make a submission file
        train_df = du.create_train_df(DATA_PATH)
        test_df = pd.read_csv(SPECTROGRAMS_TEST)
        train_df = pd.read_csv(SPECTROGRAMS_TRAIN)
        train_df['id']=train_df.label.apply(lambda x: np.nonzero(np.array(x.replace('\n','').replace('[','').replace(']','').split(' '), dtype=np.int16))[0][0])
        id_label=train_df.groupby(['id','str_label'])['label'].count()
        labels = id_label.reset_index().str_label.tolist()
        # test_df = pd.read_csv(DATA_PATH + 'sample_submission.csv')
        # test_df = test_df[['filename']].copy()
        # labels = list(train_df.label.unique())
        top_3 = np.array(labels)[np.argsort(-predictions, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test_df['label'] = predicted_labels
        add_df={'filename':['0b0427e2.npy', '6ea0099f.npy', 'b39975f5.npy'],'label':['Gong Microwave_oven Burping_or_eructation','Gong Microwave_oven Burping_or_eructation','Gong Microwave_oven Burping_or_eructation']}
        df_to_append=pd.DataFrame(add_df)
        test_df=test_df.append(df_to_append)
        test_df['filename']=test_df['filename'].str.split('.').str.get(0) + '.wav'
        test_df.rename(columns={'filename': 'fname'}, inplace=True)
        test_df.to_csv(f'../submissions/submission_train_val_{model_name}.csv',
                       index=False)


def train_for_submission(model_name, data_config, model_config, augment=False):

    train_df = du.create_train_df(DATA_PATH)

    try:
        model = getattr(models, f'get_{model_name}')(model_config, data_config)
    except AttributeError:
        raise ValueError(f'There is no creation function for {model_name}')

    if not augment:
        train_generator = du.make_train_generator(train_df,
                                                  DATA_PATH,
                                                  data_config)
    else:
        sources_df = splitting_dataset(SPECTROGRAMS_TRAIN, full_submisison=True,
                                       valid_size=1500)
        print(sources_df.head())
        sources_train = build_sources_from_metadata(sources_df,
                                                    PRE_DATA_TRAIN,
                                                    mode='train',
                                                    label_type='id')
        print(sources_train[:10])
        train_set = pd.DataFrame(sources_train, columns=['fname', 'label_idx'])
        print(train_set.head())
        train_set = train_set.set_index('fname')

        train_generator = du.make_train_generator(train_set,
                                                  DATA_PATH,
                                                  data_config, augment=True)

    directory = f'../models/{model_name}/'

    history = model.fit_generator(train_generator,
                                  epochs=model_config.max_epochs,
                                  use_multiprocessing=False, workers=1,
                                  max_queue_size=10)

    with open(directory + 'history_full.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    model.save_weights(directory + 'best_for_submission.h5')


def make_submission(model_name, data_config, model_config, augment=False):

    directory = f'../models/{model_name}/'

    if not os.path.exists(directory):
        os.mkdir(directory)

    train_for_submission(model_name, data_config, model_config, augment=augment)

    try:
        model = getattr(models, f'get_{model_name}')(model_config, data_config)
    except AttributeError:
        raise ValueError(f'There is no creation function for {model_name}')
    model.load_weights(directory + 'best_for_submission.h5')

    if not augment:
        test_generator = du.make_pred_generator(DATA_PATH,
                                                data_config)
    else:
        test_generator = du.make_pred_generator(DATA_PATH,
                                                data_config,
                                                augment=True)

    predictions = model.predict_generator(test_generator, verbose=1)
    # predictions = model.predict_generator(test_generator, verbose=1)

    # Save predictions
    np.save(directory + "raw__full_predictions.npy", predictions)

    # Make a submission file
    train_df = du.create_train_df(DATA_PATH)
    test_df = pd.read_csv(SPECTROGRAMS_TEST)
    train_df = pd.read_csv(SPECTROGRAMS_TRAIN)
    train_df['id']=train_df.label.apply(lambda x: np.nonzero(np.array(x.replace('\n','').replace('[','').replace(']','').split(' '), dtype=np.int16))[0][0])
    id_label=train_df.groupby(['id','str_label'])['label'].count()
    labels = id_label.reset_index().str_label.tolist()
    # test_df = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    # test_df = test_df[['filename']].copy()
    # labels = list(train_df.label.unique())
    top_3 = np.array(labels)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_df['label'] = predicted_labels
    add_df={'filename':['0b0427e2.npy', '6ea0099f.npy', 'b39975f5.npy'],'label':['Gong Microwave_oven Burping_or_eructation','Gong Microwave_oven Burping_or_eructation','Gong Microwave_oven Burping_or_eructation']}
    df_to_append=pd.DataFrame(add_df)
    test_df=test_df.append(df_to_append)
    test_df['filename']=test_df['filename'].str.split('.').str.get(0) + '.wav'
    test_df.rename(columns={'filename': 'fname'}, inplace=True)
    test_df.to_csv(f'../submissions/submission_full_{model_name}.csv',
                   index=False)


def splitting_dataset(data_dir, full_submisison=False, valid_size=1500):
    """
    ONLY FOR TRAIN
    Prepare the dataset creating a metadata.csv file with the label,
    image name and if train or test.
    If valid sizee >3710 then all will be for training
    Input:
        data_dir: Directory of SPECTROGRAMS_TRAIN csv
    """
    sources_df = pd.read_csv(data_dir)
    total_rows = sources_df.shape[0]
    total_verified_original = ((sources_df['verified'] == 1) &
                               (sources_df['original'] == 1)).sum()
    print('total_verified_original', total_verified_original)
    print('total_rows', total_rows)
    sources_df['split'] = 'train'
    if full_submisison:
        return sources_df
    if total_verified_original > valid_size:
        sources_df_verified_original = sources_df[(sources_df['verified'] == 1) &
                                                  (sources_df['original']==1)]
        _, sources_df_verified_original_valid = train_test_split(sources_df_verified_original,
                                                                 test_size=valid_size,
                                                                 stratify=sources_df_verified_original['str_label'])
        # print(sources_df_verified_original_valid)
        valid_names=sources_df_verified_original_valid.filename.str.split('.').str.get(0)
        sources_df.loc[sources_df_verified_original_valid.index, 'split'] = 'valid'
        sources_df['or_names']=sources_df.filename.str.split('.').str.get(0).str.split('_').str.get(0)
        sources_df=sources_df[~((sources_df['or_names'].isin(valid_names)) & (sources_df['split']=='train'))]
        # print(sources_df.head(5))
        # print(sources_df.shape)
    else:
        print('Error: Bad handling with valid size. All will be train split')
    return sources_df


def build_sources_from_metadata(metadata, data_dir, mode='train',
                                label_type='id'):
    """
    Description:
      Build sources (to fill) from the sources_df and according
      to train or valid
    Input:
        mode: Either train, valid or test
        data dir: Directory where files are located
        label_type: Can be 'str' or 'array'
    Output:
        sources: A list of tuples with the filepath and label of each
                 image if mode is either train or valid
                A list of tuples with filepath for test
    """
    df = metadata.copy()
    if mode == 'train' or mode == 'valid':
        df = df[df['split'] == mode]
        df = df.sample(frac=1)
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))
        if label_type == 'array':
            sources = list(zip(df['filepath'], df['label']))
        elif label_type == 'str':
            sources = list(zip(df['filepath'], df['str_label']))
        elif label_type == 'id':
            df['id'] = df.label.apply(lambda x: np.nonzero(np.array(x.replace('\n','').replace('[','').replace(']','').split(' '), dtype=np.int16))[0][0])
            sources = list(zip(df['filepath'], df['id']))
    elif mode == 'test':
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))
        sources = list(zip(df['filepath']))
    return sources


tf.keras.backend.clear_session()
model_name = 'aug_double_convs_wider'
data_config = du.DataConfig(augment=True)
model_config = models.ModelConfig(max_epochs=17)

# train_with_val(model_name, data_config, model_config, make_submission=True, augment=True)
make_submission(model_name, data_config, model_config, augment=True)
