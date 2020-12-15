import pandas as pd

import numpy as np
from skimage.feature import hog


def load_data():  #loading data
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


if __name__ == '__main__':
    # starting loading data
    test, train = load_data()
    # end of data load and start of dividing data into features and classes
    X_train = train[:, 1:]
    y_train = train[:, 0]
    X_test = test[:, 1:]
    y_test = test[:, 0]
    # end of data dividing and start of data transforming into form that will allow use of ML methods
    X_train_SVM = preprocess_svm(X_train)
    X_test_SVM = preprocess_svm(X_test)
    X_train = preprocess_cnn(X_train)
    X_test = preprocess_cnn(X_test)
    #
