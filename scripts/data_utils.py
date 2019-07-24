import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence


def audio_norm(data):
    """Normalize raw audio

    Arguments:
        data {np.array} -- array with audio timeseries

    Returns:
        np.array -- normalize audio timeseries
    """
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5


class DataConfig(object):
    """Class to control configurations for creating data generator.
    """
    def __init__(self,
                 sampling_rate=44100, audio_duration=2, n_classes=41,
                 use_mfcc=True, n_mfcc=20, verified_only=False):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.verified_only = verified_only

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc,
                        1 + int(np.floor(self.audio_length/512)),
                        1)
        else:
            self.dim = (self.audio_length, 1)


class DataGenerator(Sequence):
    """Class for creating a data generator that can be used with Keras models

    Arguments:
        Sequence {tf.keras.utils} -- Sequence object type

    Returns:
        tf.data.Dataset -- Data generator that can be used with fit_generator
    """
    def __init__(self, config, data_dir, list_IDs, labels=None,
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def normalize_duration(self, data):

        input_length = self.config.audio_length
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data,
                          (offset, input_length - len(data) - offset),
                          "constant")
        return data

    def __data_generation(self, list_IDs_temp):
        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, *self.dim))

        for i, ID in enumerate(list_IDs_temp):
            filepath = self.data_dir + ID

            # Read and Resample the audio
            data, _ = librosa.core.load(filepath,
                                        sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')

            # Random offset / Padding
            data = self.normalize_duration(data)

            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                            n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i, ] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X


def create_train_df(data_dir):

    df = pd.read_csv(data_dir + 'train.csv')
    df = df.set_index('fname')

    unique_labels = list(df.label.unique())
    label_idx = {label: i for i, label in enumerate(unique_labels)}
    df["label_idx"] = df.label.apply(lambda x: label_idx[x])

    return df


def make_train_generator(df, data_dir, config, training=True):
    """Complete all the steps to read file paths and create generator for
    training and validation

    Arguments:
        df {pd.DataFrame} -- dataframe with filenames and label indexes
        data_dir {str} -- data directory
        config {DataConfig} -- class with configurations for generator

    Returns:
        tf.data.Dataset -- Data generator that can be used with fit_generator
    """

    if config.verified_only:
        df = df[df.manually_verified == 1]

    if training:
        generator = DataGenerator(config, data_dir + 'audio_train/',
                                  df.index,
                                  df.label_idx, batch_size=64,
                                  preprocessing_fn=audio_norm)
    else:
        generator = DataGenerator(config, data_dir + 'audio_train/',
                                  df.index, batch_size=128,
                                  preprocessing_fn=audio_norm)

    return generator


def make_pred_generator(data_dir, config):
    """Complete all the steps to read file paths and create generator
    for prediction

    Arguments:
        data_dir {str} -- data directory
        config {DataConfig} -- class with configurations for generator

    Returns:
        tf.data.Dataset -- Data generator that can be used with fit_generator
    """

    df = pd.read_csv(data_dir + 'sample_submission.csv')
    df = df.set_index('fname')
    generator = DataGenerator(config, data_dir + 'audio_test/',
                              df.index,
                              batch_size=64,
                              preprocessing_fn=audio_norm)

    return generator
