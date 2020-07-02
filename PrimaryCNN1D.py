# -*- coding: utf-8 -*-
"""
Primary Bacterial 1D CNN
The purpose of this file is to identify correctly a bacteria strain using
a 1D CNN. The code used is a 30-class set, with 4000 spectra per class.
Created on Mon Jun 29 16:09:11 2020
@author: Kaitlyn Kukula
Reference Code: 
"""

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils, to_categorical
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
#%matplotlib inline

from keras import Sequential
from keras.layers import Conv1D, Dropout, MaxPooling1D, GlobalMaxPooling1D

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from statistics import mean, stdev

# load datasets

# def load_datasets(prefix = ''):
#     """
#     Loads datasets from directory 
#     """

max_features = 10000
max_len = 1000

trainX = np.load('X_reference.npy')
trainy = np.load('y_reference.npy')
print(trainX.shape, trainy.shape)

testX = np.load('X_test.npy')
testy = np.load('y_test.npy')
print(testX.shape, testy.shape)

# wave_num = np.load('wavenumber.npy')
# print(wave_num.shape)

# one hot encode y
trainy = to_categorical(trainy)
testy = to_categorical(testy)

    
# def evaluate_model(trainX, trainy, testX, testy):
verbose, epochs, batch_size = 0, 10, 32
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[1], trainy.shape[1]
model = Sequential()
model.add(layers.Embedding(max_features, 
                           128, 
                           input_length = max_len))
model.add(Conv1D(filters = 64, 
                 kernel_size = 7,
                 activation= 'relu',
                 input_shape = (n_timesteps, n_features)))
model.add(MaxPooling1D(5))
model.add(Conv1D(filters = 32,
                 kernel_size = 7,
                 activation = 'relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(30))

#model.add(Dense(n_outputs,
               # activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.summary()

# fit network
model.fit(trainX, trainy, 
          epochs = epochs,
          batch_size = batch_size)

# evaluate model
_, accuracy = model.evaluate(testX, testy,
                             batch_size = batch_size,
                             verbose = 0)

#history = cnn.fit_generator




"""
# summarize results

#def summarize_results(scores):
print(scores)
m, s = mean(scores), std(scores)
print('Accuracy: %.3f%% (+/-%.3f' % (m,s))
    
#def run_experiment(repeats = 10, trainX, trainy, testX, testy):
scores = list()
for r in range(repeats):
    score = evaluate_model(trainX, trainy, testX, testy)
    score = score * 100.0
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)
summarize_results(scores)
     
run_experiment()
    
    

max_features = 4000
max_len = 1000

X_train = np.load('X_reference.npy')
y_train = np.load('y_reference.npy')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')


# Create Model

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length = max_len))
model.add(layers.Conv1D(32, 5, activation = 'relu'))
model.add(layers.MaxPooling1D())

model.add(layers.Embedding(max_features, 128, input_length = max_len))
model.add(layers.Conv1D(32, 5, activation = 'relu'))
model.add(layers.MaxPooling1D())
model.add(layers.Dense(30))
 
model.add(layers.Conv1D(32, 5, activation = 'sigmoid'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer = RMSprop(lr = 1e-4),
              loss = 'binary_crossentropy',
              metrics = ['acc'])
history = model.fit(X_train, y_train,
                    epochs = 3,
                    batch_size = 128,
                    validation_split = 0.2)


# Create Performance Visualizations

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')  
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')  
plt.xlabel('epoch')
plt.legend()
plt.show()
"""









