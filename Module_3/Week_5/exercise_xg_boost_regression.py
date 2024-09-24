import pandas as pd
import numpy as np

# Call out dataset
dataset_path = "./data1.csv"
df = pd.read_csv(dataset_path)

lambda_value = 0
learning_rate = 0.3


def compute_mean(data):
    return np.mean(data)


def compute_similarity_score(data, f_value):
    sum_of_residual = 0
    n = len(data)
    for i in range(0, n):
        sum_of_residual += data[i] - f_value

    sum_of_residual_squared = sum_of_residual**2
    similarity_score = sum_of_residual_squared/(n+lambda_value)
    return similarity_score


def split_by_threshold(data, threshold):
    group0 = []
    group1 = []
    data_x = data['x'].values
    data_y = data['y'].values

    for row in range(0, len(data_x)):
        if data_x[row] < threshold:
            group0.append(data_y[row])
        else:
            group1.append(data_y[row])

    return group0, group1


def compute_gain(groups, f_value, sc_root):
    gain = 0
    similarity_score = 0

    for group in groups:
        similarity_score += compute_similarity_score(
            data=group, f_value=f_value)

    gain = similarity_score - sc_root
    return gain


def predict_output(groups, x, f_value):
    output = 0

    group_1, group_2 = groups

    for i in range(len(df['x'])):
        if x > df['x'][i]:
            if df['y'][i] in group_1:
                output = (sum(group_1) - f_value*len(group_1))/len(group_1)
                break
            else:
                output = (sum(group_2) - f_value*len(group_2))/len(group_2)
                break

    predicted_value = f_0 + learning_rate*output
    print(predicted_value)


if __name__ == "__main__":

    f_0 = compute_mean(df['y'])

    sc_root = compute_similarity_score(data=df['y'], f_value=f_0)

    # case i
    groups_1 = split_by_threshold(data=df, threshold=23.5)
    gain_1 = compute_gain(groups_1, f_0, sc_root)

    # case ii
    groups_2 = split_by_threshold(data=df, threshold=25)
    gain_2 = compute_gain(groups_2, f_0, sc_root)

    # case iii
    groups_3 = split_by_threshold(data=df, threshold=26.5)
    gain_3 = compute_gain(groups_3, f_0, sc_root)

    max_gain = max(np.array([gain_1, gain_2, gain_3]))

    # in case of x = 25
    if max_gain == gain_1:
        predict_output(groups_1, 25, f_0)

    elif max_gain == gain_2:
        predict_output(groups_2, 25, f_0)

    else:
        predict_output(groups_3, 25, f_0)
