#!/usr/bin/env python
# coding: utf-8




# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pathlib import Path
import IPython.display as ipd
import sys, os, os.path
from scipy.io import wavfile
from scipy import signal
from scipy.io import wavfile

#for loading and visualizing audio files
import librosa
import librosa.display

#to play audio
import IPython.display as ipd
from subprocess import check_output
#Keras
import keras

import warnings
warnings.filterwarnings('ignore')

audio_fpath = "C:/Users/erran/Downloads/rooster_challenge/rooster_challenge/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))


# ### Load audio file and visualize its waveform (using librosa)

x, sr = librosa.load(audio_fpath+audio_clips[0])
print(type(x), type(sr))
print(x.shape, sr)
# x is nothing but it is an audio time series


# Here we are just finding the or ploting the graph of this audio file and find the which voice is long.


plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)


# ### Convert the audio waveform to spectrogram


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')


### Reading the CSV file
data1=pd.read_csv("C:/Users/erran/Downloads/ESC-50-master/ESC-50-master/meta/esc50.csv")

data1['target'] = data1['category'].apply(lambda x: 1 if x == 'rooster' else 



#Lets as a Traget variable.
y = data1[['target']]


X = np.array(data1.drop(columns=['target','filename','category','take']), dtype = float)


input_filename = x
if input_filename[-3:] != 'wav':
    print('WARNING!! Input File format should be *.wav')
    sys.exit()

samrate, data = wavfile.read(str(input_filename))
print('Load is Done! \n')

wavData = pd.DataFrame(data)


if len(wavData.columns) == 1:
    print('Mono .wav file\n')
    wavData.columns = ['M']

    wavData.to_csv(str("_Output_mono.csv"), mode='w')

    print('Save is done ' + str(input_filename[:-4]) + '_Output_mono.csv')

else:
    print('Mono1 channel .wav file\n')
    print( len(wavData.columns))
    wavData.to_csv(str(input_filename[:-4] + "Output_Mono1_channel.csv"), mode='w')

    print('Save is done ' + str(input_filename[:-4]) + 'Output_Mono1_channel.csv')


# And the split our dataset in train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Apply VGGNet model for the classification


from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append('/home/hudi/anaconda2/lib/python2.7/site-packages/h5py')
sys.path.append('/home/hudi/anaconda2/lib/python2.7/site-packages/Keras-2.0.6-py2.7.egg')


from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras import backend as K

from sound import vggish_params2 as params


# weight path
WEIGHTS_PATH = 'C:/Users/erran/Downloads/rooster_challenge/weights/vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP = 'C:/Users/erran/Downloads/rooster_challenge/weights/vggish_audioset_weights.h5'

def VGGish(load_weights=True, weights='audioset',
           input_tensor=None, input_shape=None,
           out_dim=None, include_top=True, pooling='avg'):
    '''
    An implementation of the VGGish architecture.
    :param load_weights: if load weights
    :param weights: loads weights pre-trained on a preliminary version of YouTube-8M.
    :param input_tensor: input_layer
    :param input_shape: input data shape
    :param out_dim: output dimension
    :param include_top:whether to include the 3 fully-connected layers at the top of the network.
    :param pooling: pooling type over the non-top network, 'avg' or 'max'
    :return: A Keras model instance.
    '''

    if weights not in {'audioset', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `audioset` '
                         '(pre-training on audioset).')

    if out_dim is None:
        out_dim = params.EMBEDDING_SIZE

    # input shape
    if input_shape is None:
        input_shape = (params.NUM_FRAMES, params.NUM_BANDS, 1)

    if input_tensor is None:
        aud_input = Input(shape=input_shape, name='input_1')
    else:
        if not K.is_keras_tensor(input_tensor):
            aud_input = Input(tensor=input_tensor, shape=input_shape, name='input_1')
        else:
            aud_input = input_tensor



    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)



    if include_top:
        # FC block
        x = Flatten(name='flatten_')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
        x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
        x = Dense(out_dim, activation='relu', name='vggish_fc2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)


    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    # Create model.
    model = Model(inputs, x, name='VGGish')


    # load weights
    if load_weights:
        if weights == 'audioset':
            if include_top:
                model.load_weights(WEIGHTS_PATH_TOP)
            else:
                model.load_weights(WEIGHTS_PATH)
        else:
            print("failed to load weights")

    return model

#model save
model.save('rooster.h5')


model.compile(optimizer='adam'
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']


history = model.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=16

test_loss, test_acc = model.evaluate(X_test,y_test)

print('test_acc: ',test_acc)

x_val = X_train[:300]
partial_x_train = X_train[300:]

y_val = y_train[:300]
partial_y_train = y_train[300:]
