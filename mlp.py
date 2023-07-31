import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def init_params():
    W1 = np.random.rand(10, 30)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    W3 = np.random.rand(2, 10)
    b3 = np.random.rand(2, 1)
    return W1, b1, W2, b2, W3, b3


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return A1, Z1, A2, Z2, A3


def one_hot_encoder(Y):
    if Y == "B":
        one_hot_Y = np.array([0, 1])
    else:
        one_hot_Y = np.array([1, 0])
    return one_hot_Y.T


def deriv_ReLU(Z):
    return Z > 0


def back_prop(A1, Z1, A2, Z2, W2, A3, W3, X, Y):
    m = Y.size
    dZ3 = A3 - one_hot_encoder(Y)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    b3 = b3 - learning_rate * db3
    W3 = W3 - learning_rate * dW3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
    return np.argmax(A3, 0)


def get_accuracy(predictions, Y):
    predictions = np.where(predictions == 0, 'M', 'B')
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, learning_rate):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        A1, Z1, A2, Z2, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(A1, Z1, A2, Z2, W2, A3, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
        if i % 20 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A3), Y))
    return W1, b1, W2, b2, W3, b3


def scaling(X):
    mean = np.mean(X)
    std = np.std(X)
    normalized_arr = (X - mean) / std
    return normalized_arr


def main():
    if len(sys.argv) != 2:
        print("usage: mlp.py [dataset]")
    else:
        try:
            data = pd.read_csv(sys.argv[1])
            data.dropna(inplace=True)
            data = data.sample(frac=1).reset_index(drop=True)

            train_size = int(len(data) * 0.8)
            data_train = data[:train_size]
            data_test = data[train_size:]

            data_train = data_train.drop(axis=1, columns=["Index", "Id"])
            data_train = np.array(data_train.T)
            Y_train = data_train[0]
            X_train = data_train[1:]
            X_train_scaled = scaling(X_train)
            print(X_train_scaled)

            W1, b1, W2, b2, W3, b3 = gradient_descent(X_train_scaled, Y_train, 200, 0.1)

        except FileNotFoundError:
            print(sys.argv[1] + " not found.")


if __name__ == "__main__":
    main()