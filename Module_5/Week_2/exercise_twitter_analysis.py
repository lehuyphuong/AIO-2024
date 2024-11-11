from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from exercise_creditcard_analysis import softmax, predict, compute_loss, compute_gradient, compute_accuracy
import re
import nltk
nltk.download('stopwords')


dataset_path = 'Twitter_Data.csv'
df = pd.read_csv(
    dataset_path
)
df = df.dropna()


def text_normalize(text):
    # lowercase
    text = text.lower()

    # Retweet old acronym "RT" removal
    text = re.sub(r'^rt[\s]+', '', text)

    # Hyperlink removal
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # Punctuation removal
    text = re.sub(r'[^\w\s]', '', text)

    # Hashtags removal
    text = re.sub(r'#', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    # Stemming
    stemmer = SnowballStemmer('English')
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)

    return text


# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text']).toarray()

# add bias in X
intercept = np.ones((
    X.shape[0], 1)
)

X_b = np.concatenate(
    (intercept, X),
    axis=1
)

# One-hot encoding
n_classes = df['category'].nunique()
n_samples = df['category'].size

y = df['category'].to_numpy() + 1
y = y.astype(np.int8)
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
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
    test_size=test_size,
    random_state=random_state
)

# Define softmax function


def compute_softmax(z):

    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1)[:, None]

# Define loss fucntion


def compute_loss(y_hat, y):
    n = y.size
    return (-1/n)*np.sum(y*np.log(y_hat))

# Define predict function


def predict(X, theta):
    z = np.dot(X, theta)
    y_hat = compute_softmax(z)
    return y_hat

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
epochs = 200
batch_size = X_train.shape[0]
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
ax[0, 0].set_title('Training loss for twitter exercise')

ax[0, 1].plot(val_losses, 'red')
ax[0, 1].set(xlabel='Epoch', ylabel='loss')
ax[0, 1].set_title('validation loss for twitter exercise')

ax[1, 0].plot(train_accs)
ax[1, 0].set(xlabel='Epoch', ylabel='Acc')
ax[1, 0].set_title(' Training accuracy for twitter exercise')

ax[1, 1].plot(val_accs, 'red')
ax[1, 1].set(xlabel='Epoch', ylabel='Acc')
ax[1, 1].set_title(' Validation accuracy for twitter exercise')

plt.show()
