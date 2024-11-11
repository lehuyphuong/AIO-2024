import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset_path = 'creditcard.csv'
df = pd.read_csv(
    dataset_path
)

dataset_arr = df.to_numpy()

X, y = dataset_arr[:, :-
                   1].astype(np.float64), dataset_arr[:, -1].astype(np.uint8)

# add bias in X
intercept = np.ones((
    X.shape[0], 1)
)

X_b = np.concatenate(
    (intercept, X),
    axis=1
)

# apply one-hot encoding
n_classes = np.unique(y, axis=0).shape[0]
n_samples = y.shape[0]
# print(n_samples)
y_encoded = np.array(
    [np.zeros(n_classes) for _ in range(n_samples)]
)

y_encoded[np.arange(n_samples), y] = 1

# split training dataset
val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X_b, y_encoded,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)

# Normalize X
normalizer = StandardScaler()
X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

# Define softmax function


def softmax(z):

    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1)[:, None]

# Define predict function


def predict(X, theta):
    z = np.dot(X, theta)
    y_hat = softmax(z)
    return y_hat

# Define loss fucntion


def compute_loss(y_hat, y):
    n = y.size
    return (-1/n)*np.sum(y*np.log(y_hat))

# Define gradient function


def compute_gradient(X, y, y_hat):
    n = y.size
    return np.dot(X.T, (y_hat-y))/n

# Define updating parameters funtion


def update_theta(theta, gradient, lr):
    return theta - lr*gradient

# Define accuracy


def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta)
    acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean()

    return acc


# Train model
lr = 0.01
epochs = 30
batch_size = 1024
n_features = X_train.shape[1]

np.random.seed(random_state)
theta = np.random.uniform(
    size=(n_features, n_classes)
)

train_accs = []
train_losses = []
val_accs = []
val_losses = []
for epoch in range(epochs):
    train_batch_losses = []
    train_batch_accs = []
    val_batch_losses = []
    val_batch_accs = []

    for i in range(0, X_train.shape[0], batch_size):
        X_i = X_train[i: i + batch_size]
        y_i = y_train[i: i + batch_size]

        y_hat = predict(X_i, theta)

        train_loss = compute_loss(y_hat, y_i)

        gradient = compute_gradient(X_i, y_i, y_hat)

        theta = update_theta(theta, gradient, lr)

        train_batch_losses.append(train_loss)

        train_acc = compute_accuracy(X_train, y_train, theta)
        train_batch_accs.append(train_acc)

        y_val_hat = predict(X_val, theta)
        val_loss = compute_loss(y_val_hat, y_val)
        val_batch_losses.append(val_loss)

        val_acc = compute_accuracy(X_val, y_val, theta)
        val_batch_accs.append(val_acc)

    train_batch_loss = sum(train_batch_losses)/len(train_batch_losses)
    val_batch_loss = sum(val_batch_losses)/len(val_batch_losses)
    train_batch_acc = sum(train_batch_accs)/len(train_batch_accs)
    val_batch_acc = sum(val_batch_accs)/len(val_batch_accs)

    train_losses.append(train_batch_loss)
    val_losses.append(val_batch_loss)
    train_accs.append(train_batch_acc)
    val_accs.append(val_batch_acc)

    print(f'\n epoch {epoch + 1}: \t training loss: {
          train_batch_loss:.3} \t validation loss:{val_batch_loss:.3}')

# Virtualize result with matplotlib
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(train_losses)
ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
ax[0, 0].set_title('Training loss for creditcard exercise')

ax[0, 1].plot(val_losses, 'orange')
ax[0, 1].set(xlabel='Epoch', ylabel='loss')
ax[0, 1].set_title('validation loss for creditcard exercise')

ax[1, 0].plot(train_accs)
ax[1, 0].set(xlabel='Epoch', ylabel='Acc')
ax[1, 0].set_title(' Training accuracy for creditcard exercise')

ax[1, 1].plot(val_accs, 'orange')
ax[1, 1].set(xlabel='Epoch', ylabel='Acc')
ax[1, 1].set_title(' Validation accuracy for creditcard exercise')

plt.show()
