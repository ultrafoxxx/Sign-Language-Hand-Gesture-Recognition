
import pandas as pd


def load_data():
    test_data = pd.read_csv("sign_mnist_test.csv")
    train_data = pd.read_csv("sign_mnist_train.csv")
    print(test_data.head())
    print(train_data.head())
    return test_data, train_data


if __name__ == '__main__':
    load_data()


