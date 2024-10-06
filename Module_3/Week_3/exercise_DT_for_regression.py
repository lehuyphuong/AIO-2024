import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

ATTRIBUTE_AGE = 'age'
ATTRIBUTE_LIKES_ENGLISH = 'likes english'
ATTRIBUTE_LIKES_AI = 'likes AI'
ATTRIBUTE_RAISE_SALARY = 'raise salary'

data = {
    ATTRIBUTE_AGE: np.array([23, 25, 27, 29, 29]),
    ATTRIBUTE_LIKES_ENGLISH: np.array([0, 1, 1, 0, 0]),
    ATTRIBUTE_LIKES_AI: np.array([0, 1, 0, 1, 0]),
    ATTRIBUTE_RAISE_SALARY: np.array([200, 400, 300, 500, 400])
}

data = pd.DataFrame(data, columns=data.keys())


def compute_sse(input_data):
    n = len(input_data)
    avg = np.average(input_data)

    sse = 0
    for i in range(0, n):
        sse += (1/n)*(input_data[i] - avg)**2

    return sse


def split_compute_sse(input_data, threshold):
    group1 = []
    group2 = []
    total_sse = 0

    # split group
    for i in range(0, len(input_data)):
        if input_data[i] <= threshold:
            group1.append(data[ATTRIBUTE_RAISE_SALARY].values[i])

        else:
            group2.append(data[ATTRIBUTE_RAISE_SALARY].values[i])

    print(group1)
    print(group2)
    # compute SSE(D) = SEE(D1) + SEE(D2)
    total_sse += compute_sse(group1)
    total_sse += compute_sse(group2)

    return total_sse


if __name__ == "__main__":
    # exercise 9
    print(f"SSE of attribute 'likes AI' : {split_compute_sse(
        data[ATTRIBUTE_LIKES_AI].values, threshold=0)}")

    # exercise 10
    print(f"SSE of attribute 'age' : {split_compute_sse(
        data[ATTRIBUTE_AGE].values, threshold=24)}")

    # exercise 11
    # load dataset
    machine_cpu = fetch_openml(name='machine_cpu')
    machine_data = machine_cpu.data
    machine_labels = machine_cpu.target

    # split train:test = 8:2
    X_train, X_test, y_train, y_test = train_test_split(
        machine_data, machine_labels,
        test_size=0.2,
        random_state=42
    )

    # define model
    tree_reg = DecisionTreeRegressor(
        random_state=42,
        ccp_alpha=0.0
    )

    # train
    tree_reg.fit(X_train, y_train)

    # predict and evaluate
    y_pred = tree_reg.predict(X_test)

    print(mean_squared_error(y_test, y_pred))
