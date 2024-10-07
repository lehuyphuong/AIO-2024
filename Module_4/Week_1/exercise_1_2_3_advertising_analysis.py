import numpy as np
import matplotlib.pyplot as plt
import random


def get_column(data, index):
    result = [row[index] for row in data]

    return result


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()

    tv_data = get_column(data, 0)

    radio_data = get_column(data, 1)

    newspaper_data = get_column(data, 2)

    sale_data = get_column(data, 3)

    # buiding X input and y output for training

    X = [tv_data, radio_data, newspaper_data]

    y = sale_data

    return X, y


def initialize_params():
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)
    b = 0
    return w1, w2, w3, b


def predict(x1, x2, x3, w1, w2, w3, b):
    return x1*w1 + x2*w2 + x3*w3 + b


def compute_loss_mse(y, y_hat):
    return ((y_hat - y)**2)


def compute_loss_abs(y, y_hat):
    return abs(y_hat-y)


def compute_gradient_wi(xi, y, y_hat):
    return 2 * xi * (y_hat-y)


def compute_gradient_b(y, y_hat):
    return 2 * (y_hat-y)


def update_weight_wi(wi, dl_dwi, lr):
    return wi - lr*dl_dwi


def update_bias(b, dl_db, lr):
    return b - lr*dl_db


def implement_linear_regression(x_data, y_data, epoch_max=50, lr=1e-5, loss_func='mae', mini_batch=False):
    losses = []
    w1, w2, w3, b = initialize_params()

    N = len(y_data)

    for _ in range(epoch_max):
        dw1_total = 0
        dw2_total = 0
        dw3_total = 0
        db_total = 0
        loss_total = 0
        for i in range(N):

            # get a sample
            x1 = x_data[0][i]
            x2 = x_data[1][i]
            x3 = x_data[2][i]

            y = y_data[i]

            # compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            # compute_loss
            if loss_func == 'mse':
                loss = compute_loss_mse(y, y_hat)
            elif loss_func == 'mae':
                loss = compute_loss_abs(y, y_hat)

            # accumulate loss
            if mini_batch == True:
                loss_total = loss_total + loss

            # compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)

            dl_db = compute_gradient_b(y, y_hat)

            # accumulatee gradient
            if mini_batch == True:
                dw1_total = dw1_total + dl_dw1
                dw2_total = dw2_total + dl_dw2
                dw3_total = dw3_total + dl_dw3
                db_total = db_total + dl_db

            else:
                # update parameters
                w1 = update_weight_wi(w1, dl_dw1, lr)
                w2 = update_weight_wi(w2, dl_dw2, lr)
                w3 = update_weight_wi(w3, dl_dw3, lr)

                b = update_bias(b, dl_db, lr)
                losses.append(loss)

        # update parameters
        if mini_batch == True:
            w1 = update_weight_wi(w1, dw1_total/N, lr)
            w2 = update_weight_wi(w2, dw2_total/N, lr)
            w3 = update_weight_wi(w3, dw3_total/N, lr)

            b = update_bias(b, db_total/N, lr)

            # store logging loss
            losses.append(loss/N)

    return (w1, w2, w3, b, losses)


if __name__ == "__main__":
    X, y = prepare_data('advertising.csv')
    list_data = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
    print(list_data)

    # print(type(y))
    y_predict_test = predict(x1=1, x2=1, x3=1, w1=0, w2=0.5, w3=0, b=0.5)
    print(y_predict_test)

    l = compute_loss_mse(y_hat=1, y=0.5)
    print(l)

    g_wi = compute_gradient_wi(xi=1.0, y=1.0, y_hat=0.5)
    print(g_wi)

    g_b = compute_gradient_b(y=2.0, y_hat=0.5)
    print(g_b)

    after_wi = update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
    print(after_wi)

    after_b = update_bias(b=0.5, dl_db=-1.0, lr=1e-5)
    print(after_b)

    (w1, w2, w3, b, losses) = implement_linear_regression(
        X, y, loss_func='mae', epoch_max=100, mini_batch=False)
    print(w1, w2, w3)
    plt.plot(losses[:100])
    plt.xlabel("#iteration")
    plt.ylabel("#loss mse")
    plt.show()

    tv = 19.2
    radio = 35.9
    newspaper = 51.3
    sale = predict(x1=tv, x2=radio, x3=newspaper, w1=w1, w2=w2, w3=w3, b=b)
    print(sale)

    l = compute_loss_abs(y_hat=1, y=0.5)
    print(l)

    (w1, w2, w3, b, losses) = implement_linear_regression(
        X, y, loss_func='mse', epoch_max=1000, mini_batch=True)
    print(w1, w2, w3)
    plt.plot(losses[:1000])
    plt.xlabel("#iteration")
    plt.ylabel("#loss mse")
    plt.show()
