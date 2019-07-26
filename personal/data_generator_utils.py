# Loading some tests to verify your answers to the exercises below
# from shutil import copyfile
import tensorflow as tf
import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# # Let's work in the tensorflow's eager mode, which is more intuitive
tf.enable_eager_execution()

print(os.getcwd())
DATA_PATH=os.path.join(os.getcwd(),r'..\data')
AUDIO_TRAIN=os.path.join(DATA_PATH,'audio_train')
AUDIO_TEST=os.path.join(DATA_PATH,'audio_test')
PRE_DATA_TRAIN=os.path.join(DATA_PATH,r'preprocessed\train')
PRE_DATA_TEST=os.path.join(DATA_PATH,r'preprocessed\test')
SPECTROGRAMS_TRAIN=os.path.join(DATA_PATH,'spectrograms_df_train.csv')
SPECTROGRAMS_TEST=os.path.join(DATA_PATH,'spectrograms_df_test.csv')

#columns=['filename', 'str_label', 'verified', 'label', 'original']
def splitting_dataset(data_dir, valid_size=1500):
    """
    ONLY FOR TRAIN
    Prepare the dataset creating a metadata.csv file with the label, image name and if train or test.
    If valid sizee >3710 then all will be for training
    Input:
        data_dir: Directory of SPECTROGRAMS_TRAIN csv
    """
    sources_df = pd.read_csv(data_dir)
    total_rows = sources_df.shape[0]
    total_verified_original=((sources_df['verified']==1) & (sources_df['original']==1)).sum()
    print('total_verified_original',total_verified_original)
    print('total_rows',total_rows)
    sources_df['split']='train'
    if total_verified_original>valid_size:
        sources_df_verified_original=sources_df[(sources_df['verified']==1) & (sources_df['original']==1)]
        _,sources_df_verified_original_valid=train_test_split(sources_df_verified_original, 
                                test_size=valid_size, stratify=sources_df_verified_original['str_label'])
        sources_df.loc[sources_df_verified_original_valid.index,'split']='valid'
        print(sources_df.head(5))
    else:
        print('Error: Bad handling with valid size. All will be train split')
    
    return sources_df

def build_sources_from_metadata(metadata, data_dir, mode='train', label_type='id'):
    """
    Description:
      Build sources (to fill) from the sources_df and according to train or valid
    Input:
        mode: Either train, valid or test
        data dir: Directory where files are located
        label_type: Can be 'str' or 'array'
    Output:
        sources: A list of tuples with the filepath and label of each image if mode is either train or valid
                A list of tuples with filepath for test
    """
    df = metadata.copy()
    if mode=='train' or mode=='valid':
        df = df[df['split'] == mode]
        df=df.sample(frac=1)
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))
        if label_type=='array':
            sources = list(zip(df['filepath'], df['label']))
        elif label_type=='str':
            sources = list(zip(df['filepath'], df['str_label']))
        elif label_type=='id':
            df['id']=df.label.apply(lambda x: np.nonzero(np.array(x.replace('\n','').replace('[','').replace(']','').split(' '), dtype=np.int16))[0][0])
            sources = list(zip(df['filepath'], df['id']))
    elif mode=='test':
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))
        sources = list(zip(df['filepath']))
    return sources

def preprocess_image(image):
    image = tf.image.resize(image, size=(32, 32))
    image = image / 255.0
    return image

def augment_image(image):
    return image

def make_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources
    Input:
        sources (list): A list of (filepath, label_id) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.
    Output:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load_npy(filepath):
        # print('filepath: ........',filepath.numpy())
        npy=np.load(filepath.numpy())
        # print('npy_in',npy)
        return npy

    def load(row):
        print(row)
        filepath = row['image']
        # print(filepath)
        # img_npy = tf.decode_raw(npy, tf.float32)
        img_npy = tf.py_function(func=load_npy, inp=[filepath], Tout=tf.float32)
        # print('img_npy',img_npy)
        return img_npy, row['label']

    images, labels = zip(*sources)
    
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'label': list(labels)}) 

    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    # ds = ds.map(lambda x,y: (preprocess_image(x), y))
    
    # if training:
    #     ds = ds.map(lambda x,y: (augment_image(x), y))
        
    ds = ds.repeat(count=num_epochs)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(1)
    return ds

def imshow_batch_of_three(batch, show_label=True):
    print(batch)
    label_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img, aspect='auto', origin='lower')
        if show_label:
            axarr[i].set(xlabel='label = {}'.format(label_batch[i]))
    plt.show()


sources_df_train=splitting_dataset(SPECTROGRAMS_TRAIN, valid_size=1500)
sources_df_test=pd.read_csv(SPECTROGRAMS_TEST)
sources_train=build_sources_from_metadata(sources_df_train, PRE_DATA_TRAIN, mode='train', label_type='id')
sources_test=build_sources_from_metadata(sources_df_train, PRE_DATA_TEST, mode='test', label_type='id')
print(sources_train[:10])
dataset_train=make_dataset(sources_train, training=False, batch_size=3, num_epochs=1, num_parallel_calls=3, shuffle_buffer_size=None)
dataset_valid=make_dataset(sources_train, training=False, batch_size=3, num_epochs=1, num_parallel_calls=3, shuffle_buffer_size=None)
dataset_test=make_dataset(sources_train, training=False, batch_size=3, num_epochs=1, num_parallel_calls=3, shuffle_buffer_size=None)
dataset = iter(dataset)
print(next(dataset))
imshow_batch_of_three(next(dataset))