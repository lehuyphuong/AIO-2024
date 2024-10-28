import numpy as np
import matplotlib.pyplot as plt
import random

data = np.genfromtxt('advertising.csv', delimiter=',', skip_header=1)
N = data.shape[0]
X = data[:, :3]
y = data[:, 3:]

# Normalize input data by using mean normalization


def mean_normalization(X):
    N = len(X)
    maxi = np.max(X)
    mini = np.min(X)
    avg = np.mean(X)
    X = (X-avg)/(maxi-mini)
    X_b = np.c_[np.ones((N, 1)), X]
    return X_b, maxi, mini, avg


X_b, maxi, mini, avg = mean_normalization(X)


def stochastic_gradient_descent(X_b, y, n_epochs=50, learning_rate=0.00001):
    thetas = np.array([[1.1627037], [-0.81960489], [1.39501033], [0.29763545]])
    thetas_path = [thetas]
    losses = []

    for _ in range(n_epochs):
        for i in range(N):
            # select random number in N
            # random_index = np.random.randint(N)
            random_index = i
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # Compute output
            y_hat = xi.dot(thetas)

            # Compute loss li
            loss = (y_hat-yi)*(y_hat-yi)/2

            # Compute gradient for loss
            g_li = y_hat-yi
            gradients = xi.T.dot(g_li)

            # Update theta
            thetas = thetas - learning_rate*gradients

            # logging
            losses.append(loss[0][0])
            thetas_path.append(thetas)

    return thetas_path, losses


def mini_batch_gradient_descent(X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01):
    thetas = np.array([[1.1627037], [-0.81960489], [1.39501033], [0.29763545]])

    thetas_path = [thetas]
    losses = []

    for _ in range(n_epochs):
        shuffled_indices = np.asarray([21, 144, 17, 107, 37, 115, 167, 31, 3,
                                       132, 179, 155, 36, 191, 182, 170, 27, 35, 162, 25, 28, 73, 172, 152, 102, 16,
                                       185, 11, 1, 34, 177, 29, 96, 22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126,
                                       165, 78, 151, 104, 110, 53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 190,
                                       169, 149, 79, 138, 20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 26, 8, 131,
                                       77, 80, 130, 127, 125, 61, 10, 175, 143, 87, 33, 50, 54, 97, 9, 84, 188, 139,
                                       195, 72, 64, 194, 44, 109, 112, 60, 86, 90, 140, 171, 59, 199, 105, 41, 147,
                                       92, 52, 124, 71, 197, 163, 98, 189, 103, 51, 39, 180, 74, 145, 118, 38, 47,
                                       174, 100, 184, 183, 160, 69, 91, 82, 42, 89, 81, 186, 136, 63, 157, 46, 67,
                                       129, 120, 116, 32, 19, 187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40, 83, 24,
                                       168, 150, 178, 49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65, 176, 101, 55,
                                       133, 13, 106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2, 134, 56, 123, 122,
                                       154])

        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, N, minibatch_size):
            xi = X_b_shuffled[i: i+minibatch_size]
            yi = y_shuffled[i: i+minibatch_size]

            # Compute output
            y_hat = xi.dot(thetas)

            # Compute loss
            loss = ((y_hat-yi)**2)/2

            # Compute gradient
            loss_grd = (y_hat-yi)/minibatch_size
            gradients = xi.T.dot(loss_grd)

            # Update theta
            thetas = thetas - learning_rate*gradients

            # Log
            losses.append(np.sum(loss)/minibatch_size)
            thetas_path.append(thetas)

    return thetas_path, losses


def batch_gradient_descent(X_b, y, n_epochs=100, lr=0.01):
    thetas = np.array([[1.1627037], [-0.81960489], [1.39501033], [0.29763545]])
    thetas_path = [thetas]
    losses = []

    for _ in range(n_epochs):

        # Compute output
        y_hat = X_b.dot(thetas)

        # Compute loss
        loss = ((y_hat-y)**2)

        # Compute gradient
        loss_grad = 2*(y_hat-y)/N
        gradient = X_b.T.dot(loss_grad)

        # Update thetas
        thetas = thetas - lr*gradient

        # Log
        thetas_path.append(thetas)
        losses.append(np.sum(loss)/N)

    return thetas_path, losses


sgd_theta, losses = stochastic_gradient_descent(
    X_b, y, n_epochs=50, learning_rate=0.01)
print(round(sum(losses), 2))
x_axis = list(range(500))
plt.plot(x_axis, losses[:500])
plt.show()

mbdg_theta, losses = mini_batch_gradient_descent(
    X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01)
x_axis = list(range(200))
print(round(sum(losses), 2))
plt.plot(x_axis, losses[:200])
plt.show()


bgd_thetas, losses = batch_gradient_descent(X_b, y, n_epochs=100, lr=0.01)
x_axis = list(range(100))
print(round(sum(losses), 2))
plt.plot(x_axis, losses[:100])
plt.show()
