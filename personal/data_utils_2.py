# Loading some tests to verify your answers to the exercises below
# from shutil import copyfile
import tensorflow as tf
import os
import pandas as pd
import librosa
from sklearn import preprocessing
import random
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import random
rando.seed(10)
# Let's work in the tensorflow's eager mode, which is more intuitive
tf.enable_eager_execution()

class DataProcess():
    def __init__(self):
        # Variables
        print(os.getcwd())
        self.DATA_PATH=os.path.join(os.getcwd(),r'..\data')
        self.AUDIO_TRAIN=os.path.join(self.DATA_PATH,'audio_train')
        self.AUDIO_TEST=os.path.join(self.DATA_PATH,'audio_test')
        self.PRE_DATA_TRAIN=os.path.join(self.DATA_PATH,r'preprocessed\train')
        self.PRE_DATA_TEST=os.path.join(self.DATA_PATH,r'preprocessed\test')
        self.EPS = 1e-8
        self.THRESHOLD_SPLIT=0.8
        self.sample_rate=44100 #Sample rate of audios
        self.bit_depth=65536 # Bit depth or value that may get
        self.duration = 5 #segs
        self.nfft=2024 # n_fft window size
        self.created_train=False
        self.created_test=False
        
        # Computed
        self.input_length=self.duration*self.sample_rate # Desired input length to stft
        self.hop_length=self.nfft//4 # frame_shift
        self.n_frames=1 + (self.input_length - self.nfft)//self.hop_length #output frames   
        print('n_frames',self.n_frames)     

    def read_csv_file(self):
        # Reading csvs
        self.metadata_test=pd.read_csv(os.path.join(self.DATA_PATH,'sample_submission.csv'))
        self.metadata_train=pd.read_csv(os.path.join(self.DATA_PATH,'train.csv'))

        #Encoding labels
        self.UNIQUE_CLASSES=self.metadata_train['label'].unique()

        # Training
        # One hot encoded of train
        for class_i in self.UNIQUE_CLASSES:
            mask=self.metadata_train['label']==class_i
            self.metadata_train[class_i]=mask*1

        # Getting filepaths for each audio clip
        self.metadata_train['filepath']=self.metadata_train['fname'].apply(lambda x: os.path.join(self.AUDIO_TRAIN,x))
        #Test
        self.metadata_test['filepath']=self.metadata_test['fname'].apply(lambda x: os.path.join(self.AUDIO_TEST,x)) 

        #Create spectrogram
        print('Begining with Test')
        self.creating_spectrograms(mode='test', augment=False)
        print('Begining with Train')
        self.creating_spectrograms(mode='train', augment=True)
        # #Train and Valid splitting
        # splits = ['train' if random.random() <= self.THRESHOLD_SPLIT else 'valid' for _ in range(len(self.metadata_train))]
        # self.metadata_train['split']=splits

    def creating_spectrograms(self, mode='train', augment=False):
        """
        Description:
            Create spectrogram of each sample selected from a dataframe with filepath
        Input:
            DataFrame - DataFrame with file paths to create spectrograms and columns indicating which type of augmentation to apply
        Returns:
            Dataframe - With the filepath of each image generated
            Image or spectrogram of each file in the specified folder
        """
        # Getting DataFrame
        if mode=='train':
            if self.created_train:
                print('Train data already created. Erase it first')
                return None
            df=self.metadata_train
            path_to_save=self.PRE_DATA_TRAIN
        elif mode=='test':
            if self.created_test:
                print('Test data already created. Erase it first')
                return None
            df=self.metadata_test
            path_to_save=self.PRE_DATA_TEST
        spectrogram_df=[]
        #print('df:',df.head())
        # Audio processing
        audio_files=df['filepath'].values
        for idx, audio_file in enumerate(audio_files):
            raw_audio_wav = self.obtaining_audio(audio_file)
            if raw_audio_wav.shape[0]==0:
                print('idx, audio_file: ', (idx, audio_file))
                print('raw_audio_wav.shape: ',raw_audio_wav.shape)
                continue
            norm_audio_wav = self.normalize_n_frames(raw_audio_wav, truncate='left')
            logspect=self.get_spectrogram_stft(norm_audio_wav)
            name=os.path.split(audio_file)[1]
            name=name.split('.')[0]

            #Save img
            np.save(os.path.join(path_to_save,name), logspect)

            # # Plotting
            # plt.imshow(logspect, aspect='auto', origin='lower',vmin=-10, vmax=10)
            # plt.title('spectrogram of origin audio')
            # plt.show()
            if mode=='train':
                label=df.iloc[idx,3:-1].values
                str_label=df.iloc[idx,1]
                verified=df.iloc[idx,2]
                to_save=[(name+'.npy', str_label, verified, label, 1)] # 1 for original
                if augment:
                    augment_save=self.augment_audio(raw_audio_wav, name, str_label, verified, label, path_to_save)
                    to_save=to_save+augment_save
            elif mode=='test':
                to_save=[(name+'.npy')]
            # print('to_save',to_save)
            spectrogram_df=spectrogram_df+to_save
        
        #Creating final dataframe with all correct files
        if mode=='train':
            spectrogram_df=pd.DataFrame(spectrogram_df, columns=['filename', 'str_label', 'verified', 'label', 'original'])
            self.spectrogram_df_train=spectrogram_df
            self.created_train=True
            print(self.spectrogram_df_train)
            print('Finished train')
            self.spectrogram_df_train.to_csv(os.path.join(self.DATA_PATH,'spectrograms_df_train.csv'), index=False)

        elif mode=='test':
            spectrogram_df=pd.DataFrame(spectrogram_df, columns=['filename'])
            self.spectrogram_df_test=spectrogram_df
            self.created_test=True
            print(self.spectrogram_df_test)
            print('Finished test')
            self.spectrogram_df_test.to_csv(os.path.join(self.DATA_PATH,'spectrograms_df_test.csv'), index=False)

    def normalize_n_frames(self, raw_audio, truncate='left'):
        # Audio is larger than desired input truncate left
        if len(raw_audio) > self.input_length:
            if truncate=='left':
                normalized_frame_audio = raw_audio[:self.input_length]
            elif truncate=='right':
                max_offset = len(raw_audio) - self.input_length
                normalized_frame_audio = raw_audio[max_offset:self.input_length+max_offset]
        # Audio shorter than desired, then pad with reflect method at the right
        elif len(raw_audio) < self.input_length:
                max_offset = self.input_length - len(raw_audio)
                normalized_frame_audio = np.pad(raw_audio, (0, max_offset), mode='reflect')
        # Equal don't do anything, keep raw audio
        else:
            normalized_frame_audio=raw_audio

        return normalized_frame_audio

    def augment_audio(self, raw_wav, name, str_label, verified, label, path_to_save):
        augment_save=[]
        choices=random.choice([1,2,3,4])
        #For volume
        vol_adj = random.choice([0.8,0.85,1.15,1.2])
        #For shifting
        scale=0.0005
        shift_length=random.choice([1100, 1980]) #represents 20 ms 30 ms
        # Volume change
        if choices == 1 or choices==4:
            wav_vol=raw_wav * vol_adj
            norm_audio_wav = self.normalize_n_frames(wav_vol, truncate='left')
            logspect=self.get_spectrogram_stft(norm_audio_wav)
            #Save img
            np.save(os.path.join(path_to_save,name+'_v'), logspect)
            augment_save+=[(name+'_v'+'.npy', str_label, verified, label, 0)] #0 for augmented
            # # Plotting
            # plt.imshow(logspect, aspect='auto', origin='lower',vmin=-10, vmax=10)
            # plt.title('Volume change' + str(vol_adj))
            # plt.show()
        # Shift audio
        if choices == 2 or choices == 4:
            if self.input_length - len(raw_wav) > 220:
                if random.random() < 0.5: #right shift
                    wav_shift = np.hstack((scale*np.random.rand(shift_length), raw_wav))
                else: #left shift
                    wav_shift = np.hstack((raw_wav, scale*np.random.rand(shift_length)))
            else:
                wav_shift = np.hstack((scale*np.random.rand(shift_length), raw_wav))
            norm_audio_wav = self.normalize_n_frames(wav_shift, truncate='left')
            logspect=self.get_spectrogram_stft(norm_audio_wav)
            #Save img
            np.save(os.path.join(path_to_save,name+'_s'), logspect)
            augment_save+=[(name+'_s'+'.npy', str_label, verified, label, 0)] # 0 for augmented
            # # Plotting
            # plt.imshow(logspect, aspect='auto', origin='lower',vmin=-10, vmax=10)
            # plt.title('Shifting' + str(shift_length))
            # plt.show()
        if choices == 3:
            wav_vol = raw_wav * vol_adj
            if (self.input_length - len(wav_vol)) > 220:
                if random.random() < 0.5:
                    wav_shift = np.hstack((scale*np.random.rand(shift_length), wav_vol))
                else:
                    wav_shift = np.hstack((wav_vol, scale*np.random.rand(shift_length)))
            else:
                wav_shift = np.hstack((scale*np.random.rand(shift_length), wav_vol))
            norm_audio_wav = self.normalize_n_frames(wav_shift, truncate='left')
            logspect=self.get_spectrogram_stft(norm_audio_wav)
            #Save img
            np.save(os.path.join(path_to_save,name+'_vs'), logspect)
            augment_save+=[(name+'_vs'+'.npy', str_label, verified, label, 0)] # 0 for augmented

        return augment_save

    def get_spectrogram_stft(self, wav):
        """
        Input:
            wav: wav array from audio (np.array)
        Output:
            Short time fourier spectrogram (stft)
        Intern Parameters:
            n_fft: Fourier fast tranform window size (n_freq)
            hop_length: number audio of frames between STFT columns (Tx) (win_lenght//4)
            win_length: Each frame of audio is windowed by window(). The window will be of length win_length and then padded with zeros to match n_fft.
            window: window funtion hann (No idea)

        return:
            spec: np array(n_fft//2+1,t)
        """
        D = librosa.stft(wav, n_fft=self.nfft, center=True,
                        pad_mode = 'reflect',
                        win_length=self.nfft, window='hamming')
        spect, phase = librosa.magphase(D) # Decompose in magnitude and phase
        log_spect = np.log(spect + self.EPS)
        # print(log_spect.dtype)
        return log_spect

    def obtaining_audio(self, audio_file, mode='stft'):
        """
        Load audio from .wav into numpy array
        Input:
            audio_file: audio filepath
        Output:
            wav: Audio loaded as array
        """
        wav, sr = librosa.load(audio_file, sr=None)
        return wav

    def build_sources(self):
        """
        # Building sources (filepath, labels)
        """
        df=self.metadata_train.copy()
        df_train=df[df['split']=='train']
        df_valid=df[df['split']=='valid']

        df_test=self.metadata_test.copy()

        self.train_sources=list(zip(df_train['filepath'],df_train.iloc[:,3:-2].values))
        self.valid_sources=list(zip(df_valid['filepath'],df_valid.iloc[:,3:-2].values))
        self.test_sources=list(zip(df_test['filepath'],np.zeros((df_test.shape[0],df_valid.iloc[:,3:-2].shape[1]))))

        # # Reading .wav files 
        # wav, sr = librosa.load(self.train_sources[0][0], sr=None)
        # log_spect = np.log(self.get_spectrogram(wav))
        # print('spectrogram shape:', log_spect.shape)
        # plt.imshow(log_spect, aspect='auto', origin='lower',)
        # plt.title('Example: Spectrogram of origin audio')
        # plt.show()

        # print(type(log_spect))
        # print(type(train_sources[0][1]))
        # tf_spec=tf.convert_to_tensor(log_spect)
        # tf_label=tf.convert_to_tensor(train_sources[0][1])

    # def load(self, sample):
    #     """
    #     tf.py_func is deprecated in TF V2. Instead, use
    #     tf.py_function, which takes a python function which manipulates tf eager
    #     tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    #     an ndarray (just call tensor.numpy()) but having access to eager tensors
    #     means `tf.py_function`s can use accelerators such as GPUs as well as
    #     being differentiable using a gradient tape.
    #     """
    #     audio_file=sample['audio']
    #     label=sample['label']
    #     tf_spec=tf.py_function(self.obtaining_audio, [audio_file], tf.float32)
    #     tf_label=tf.convert_to_tensor(label)
    #     print(tf_spec)
    #     print(tf_label)
    #     return tf_spec, tf_label
    
    # def minmax(self, spectrogram):
    #     x = spectrogram
    #     x -= tf.reduce_min(x)
    #     x /= tf.reduce_max(x)
    #     return x

    # def input_fn(self, batch_size=1):
    #     interleave = tf.contrib.data.parallel_interleave
    #     return (
    #         tf.data.Dataset
    #         .list_files('piano.wav')
    #         .shuffle(32)
    #         .repeat(10)
    #         .map(load)
    #         .apply(interleave(segment, batch_size))
    #         .map(transform)
    #         .map(minmax)
    #         .batch(batch_size)
    #     )

    def creating_dataset(self, mode='train', batch_size=1, num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
        """
        Returns an operation to iterate over the dataset specified in sources
        Args:
            training (bool): whether to apply certain processing steps
                defined only in training mode (e.g. shuffle).
            batch_size (int): number of elements the resulting tensor
                should have.
            num_epochs (int): Number of epochs to repeat the dataset.
            num_parallel_calls (int): Number of parallel calls to use in
                map operations.
            shuffle_buffer_size (int): Number of elements from this dataset
                from which the new dataset will sample.
        Returns:
            A tf.data.Dataset object. It will return a tuple images of shape
            [N, H, W, CH] and labels shape [N, 1].
        """
        if mode=='train':
            audios, labels = zip(*self.train_sources[:10])
        elif mode=='valid':
            audios, labels = zip(*self.valid_sources[:10])
        elif mode=='test':
            audios,labels = zip(*self.test_sources[:10])
        print(list(audios),list(labels))
        ds = tf.data.Dataset.from_tensor_slices({'audio': list(audios), 'label': list(labels)})

        # if mode=='train':
        #     ds = ds.shuffle(shuffle_buffer_size)

        ds = ds.map(self.load)
        # self.ds = self.ds.map(self.load) # Preprocess
        #if mode=='train':
        #   self.ds = self.ds.map(self.load) # Augment (only train)
        ds = ds.repeat(count=num_epochs)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(1)
        return ds

dp=DataProcess()
dp.read_csv_file()
