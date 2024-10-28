import numpy as np
import matplotlib.pyplot as plt
import random


def get_column_value(data, index):
    result = [row[index] for row in data]

    return result


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()

    tv_data = get_column_value(data, 0)

    radio_data = get_column_value(data, 1)

    newspaper_data = get_column_value(data, 2)

    sale_data = get_column_value(data, 3)

    # buiding X input and y output for training

    X = [[1, x1, x2, x3]
         for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]

    y = sale_data

    return X, y


def initialize_params():
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)
    b = 0
    return [b, w1, w2, w3]


def predict(x_features, weights):
    return np.dot(x_features, weights)


def compute_loss(y, y_hat):
    return ((y_hat - y)**2)


def compute_gradient(xi, y, y_hat):
    return [2 * (y_hat-y), 2 * xi[1] * (y_hat-y), 2 * xi[2] * (y_hat-y), 2 * xi[3] * (y_hat-y)]


def update_weight_and_bias(wi, dl_d_b_wi, lr):
    return wi - lr*dl_d_b_wi


def implement_linear_regression(x_data, y_data, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialize_params()

    N = len(y_data)

    for _ in range(epoch_max):
        for i in range(N):

            # get a sample
            xi = x_data[i]

            y = y_data[i]

            # compute output
            y_hat = predict(xi, weights)

            # compute_loss
            loss = compute_loss(y, y_hat)

            # compute gradient w1, w2, w3, b
            dl_d_b_wi = compute_gradient(xi, y, y_hat)
            dl_d_b_wi = np.array(dl_d_b_wi, dtype=np.float32)

            # update parameters
            weights = update_weight_and_bias(
                wi=weights, dl_d_b_wi=dl_d_b_wi, lr=lr)

            losses.append(loss)

    return (weights, losses)


if __name__ == "__main__":
    X, y = prepare_data('advertising.csv')
    W, L = implement_linear_regression(X, y)

    print(L[9999])
    plt.plot(L[:100])
    plt.xlabel("#iteration")
    plt.ylabel("#loss mse")
    plt.show()
