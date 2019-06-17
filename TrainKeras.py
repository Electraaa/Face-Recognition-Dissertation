import re

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import pickle, os
import tensorflow
import pydot

K.set_image_dim_ordering('tf')

def mlp_model():
    len_files = len(os.listdir("faces/"))
    print(len_files)
    model = Sequential()
    model.add(Dense(128, input_shape=(128,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len_files, activation='softmax'))
    sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath = "mlp_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    return model, callbacks_list


def train():
    with open("train_features", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("test_features", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    model, callbacks_list = mlp_model()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=100, batch_size=50,
              callbacks=callbacks_list)
    scores = model.evaluate(test_images, test_labels, verbose=3)
    print(scores)



train()
