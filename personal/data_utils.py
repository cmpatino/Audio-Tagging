import tensorflow as tf
# Loading some tests to verify your answers to the exercises below
# from shutil import copyfile
import os
import pandas as pd
import librosa
from sklearn import preprocessing
import random
import numpy as np
import matplotlib.pyplot as plt

def get_spectrogram(wav):
    """
    wav: wav file
    n_fft: Fourier fast tranform window size (n_freq)
    hop_length: number audio of frames between STFT columns (Tx)
    win_length: Each frame of audio is windowed by window(). The window will be of length win_length and then padded with zeros to match n_fft.
    window: window funtion hann (No idea)
    """
    D = librosa.stft(wav, n_fft=480, hop_length=160,
                     win_length=480, window='hamming')
    spect, phase = librosa.magphase(D)
    return spect

# Let's work in the tensorflow's eager mode, which is more intuitive
tf.enable_eager_execution()

# Variables
print(os.getcwd())
DATA_PATH=os.path.join(os.getcwd(),r'..\data')
AUDIO_TRAIN=os.path.join(DATA_PATH,'audio_train')
AUDIO_TEST=os.path.join(DATA_PATH,'audio_test')
EPS = 1e-8
THRESHOLD_SPLIT=0.8

# Reading csvs
metadata_test=pd.read_csv(os.path.join(DATA_PATH,'sample_submission.csv'))
metadata_train=pd.read_csv(os.path.join(DATA_PATH,'train.csv'))

#Encoding labels
UNIQUE_CLASSES=metadata_train['label'].unique()

# Training
# One hot encoded
for class_i in UNIQUE_CLASSES:
    print(class_i)
    mask=metadata_train['label']==class_i
    metadata_train[class_i]=mask*1

# Getting filepaths for each audio clip
metadata_train['filepath']=metadata_train['fname'].apply(lambda x: os.path.join(AUDIO_TRAIN,x))

#Test
metadata_test['filepath']=metadata_test['fname'].apply(lambda x: os.path.join(AUDIO_TEST,x)) 
print(metadata_test.head())

#Train and Valid splitting
splits = ['train' if random.random() <= THRESHOLD_SPLIT else 'valid' for _ in range(len(metadata_train))]
metadata_train['split']=splits

# Building sources (filepath, labels)
df=metadata_train.copy()
df_train=df[df['split']=='train']
df_valid=df[df['split']=='valid']

df_test=metadata_test.copy()

train_sources=list(zip(df_train['filepath'],df_train.iloc[:,3:-2].values))
valid_sources=list(zip(df_valid['filepath'],df_valid.iloc[:,3:-2].values))
test_sources=list(zip(df_test['filepath']))

# Reading .wav files 
wav, sr = librosa.load(train_sources[0][0], sr=None)
log_spect = np.log(get_spectrogram(wav))
print('spectrogram shape:', log_spect.shape)
plt.imshow(log_spect, aspect='auto', origin='lower',)
plt.title('Example: Spectrogram of origin audio')
plt.show()

print(type(log_spect))
print(type(train_sources[0][1]))
tf_spec=tf.convert_to_tensor(log_spect)
tf_label=tf.convert_to_tensor(train_sources[0][1])

audios, labels=zip(*train_sources[:10])

def obtaining_audio(audio_file):
    wav, sr = librosa.load(audio_file, sr=None)
    log_spect = np.log(get_spectrogram(wav))
    return log_spect

def load(sample):
    audio_file=sample['audio']
    label=sample['label']
    tf_spec=tf.py_func(obtaining_audio, [audio_file], tf.float32)
    tf_label=tf.convert_to_tensor(label)
    print(tf_spec)
    print(tf_label)
    return tf_spec, tf_label

ds = tf.data.Dataset.from_tensor_slices({'audio': list(audios), 'label': list(labels)})
ds = ds.map(load)

print(ds)
# ds = ds.repeat(count=num_epochs)
# ds = ds.batch(batch_size=batch_size)
# ds = ds.prefetch(1)

print(tf_spec)
print(tf_label)
print(metadata_train.head())