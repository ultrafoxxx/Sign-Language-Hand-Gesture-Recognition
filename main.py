import pandas as pd

import numpy as np
from skimage.feature import hog
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import tensorflow


def load_data():  # loading data
    test_data = pd.read_csv("sign_mnist_test.csv")
    train_data = pd.read_csv("sign_mnist_train.csv")
    return test_data.to_numpy(), train_data.to_numpy()


def preprocess_svm(x):  # preprocessing data for svm and knn
    feature_list = []
    for image in x:
        hog_features = hog((image.reshape(28, 28)), block_norm='L2-Hys')
        feature_list.append(hog_features)
    features = np.array(feature_list)
    return features


def preprocess_cnn(x: np.ndarray):  # preprocessing data for cnn
    return x.reshape(-1, 28, 28, 1)


def create_image_generators(x_train):
    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              featurewise_center=False,
                                              samplewise_center=False,
                                              featurewise_std_normalization=False,
                                              samplewise_std_normalization=False,
                                              zca_whitening=False,
                                              rotation_range=10,
                                              zoom_range=0.1,
                                              width_shift_range=0.1,
                                              height_shift_range=0.1,
                                              horizontal_flip=False,
                                              vertical_flip=False)
    train_data_generator.fit(x_train)
    validation_generator = ImageDataGenerator(rescale=1. / 255)
    return train_data_generator, validation_generator


def create_first_model():
    model = tensorflow.keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(26, activation="softmax"))
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def create_second_model():
    model = tensorflow.keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(26, activation="softmax"))
    model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

