import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


ATTRIBUTE_AGE = 'age'
ATTRIBUTE_LIKES_ENGLISH = 'likes english'
ATTRIBUTE_LIKES_AI = 'likes AI'
ATTRIBUTE_RAISE_SALARY = 'raise salary'

data = {
    ATTRIBUTE_AGE: np.array([23, 25, 27, 29, 29]),
    ATTRIBUTE_LIKES_ENGLISH: np.array([0, 1, 1, 0, 0]),
    ATTRIBUTE_LIKES_AI: np.array([0, 1, 0, 1, 0]),
    ATTRIBUTE_RAISE_SALARY: np.array([0, 0, 1, 1, 0])
}

data = pd.DataFrame(data, columns=data.keys())


def compute_gini(input_values):
    n = input_values.sum()
    p_sum = 0

    for key in input_values.keys():
        p_sum = p_sum + (input_values[key]/n) ** 2

    gini = 1 - p_sum

    return gini

# If we split data and choose Like English as root note, we have a function


def split_by_likes_english(dataset):
    group0 = []
    group1 = []
    data_likes_english = dataset[ATTRIBUTE_LIKES_ENGLISH].values
    data_raise_salary = dataset[ATTRIBUTE_RAISE_SALARY].values

    for row in range(0, len(data_likes_english)):
        if data_likes_english[row] == 0:
            # Like English = 0
            group0.append(data_raise_salary[row])
        else:
            # Like English = 1
            group1.append(data_raise_salary[row])

    return group0, group1

# second appoarch for calculating gini impurity


def compute_gini_impurity(groups):
    gini = 0.0
    total_values = sum([len(group) for group in groups])

    for group in groups:
        size = len(group)
        if size == 0:
            continue

        p_sum = 0
        p_0 = [row for row in group].count(0) / size
        p_1 = [row for row in group].count(1) / size
        p_sum += p_0**2 + p_1**2

        # weight the gini by the size of group
        gini += (1 - p_sum) * (size/total_values)
        print(gini)

    return gini


# If we split data and choose age as root note, we have a function

def split_by_ages(dataset, threshold):
    group0 = []
    group1 = []
    data_ages = dataset[ATTRIBUTE_AGE].values
    data_raise_salary = dataset[ATTRIBUTE_RAISE_SALARY].values

    for row in range(0, len(data_ages)):
        if data_ages[row] <= threshold:
            # age <= threshold
            group0.append(data_raise_salary[row])
        else:
            # age > threshold
            group1.append(data_raise_salary[row])

    return group0, group1


# compute entropy
def compute_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts/counts.sum()
    entropy = -np.sum(probs*np.log2(probs))
    return entropy

# compute Gain


def compute_gain():
    pass


if __name__ == "__main__":

    # Exercise 2
    print(f'Gini for raise slalary : {compute_gini(data[
        ATTRIBUTE_RAISE_SALARY].value_counts())}')

    # Exercise 3
    # If we choose attribute 'Like English' as root note, we have:
    groups = split_by_likes_english(data)
    print(f'Gini likes english root node : {compute_gini_impurity(groups)}')

    # Exercise 4
    groups = split_by_ages(data, threshold=26)
    print(f'Gini for age as root node : {compute_gini_impurity(groups)}')

    # Exercise 5
    print(f'Entropy for raise slalary : {compute_entropy(data[
        ATTRIBUTE_RAISE_SALARY].value_counts())}')

    # Exercise 6
    entropy_root = compute_entropy(data[ATTRIBUTE_LIKES_ENGLISH].values)
    groups = split_by_likes_english(data)
    entropy_children = 0
    for group in groups:
        entropy_children += compute_entropy(group) * \
            (len(group)/len(data[ATTRIBUTE_LIKES_ENGLISH].values))

    print(f"Gain = : {entropy_root - entropy_children}")

    # Exercise 8
    # load the diabetes dataset
    iris_X, iris_y = datasets.load_iris(return_X_y=True)

    # split train : test
    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y,
        test_size=0.2,
        random_state=42
    )

    # define model
    dt_classifier = DecisionTreeClassifier(
        ccp_alpha=0.0,
        random_state=42)

    # train
    dt_classifier.fit(X_train, y_train)

    # predict and evaluate
    y_pred = dt_classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))
