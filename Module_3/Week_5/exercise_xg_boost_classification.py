import pandas as pd
import numpy as np

# Call out dataset
dataset_path = "./data2.csv"
df = pd.read_csv(dataset_path)

lambda_value = 0
learning_rate = 0.3
f_0 = 0.5


def compute_similarity_score(data, f_values):
    sum_of_residual = 0
    n = len(data)
    denominator_value = 0

    for i in range(n):
        sum_of_residual += data[i] - f_values[i]
        denominator_value += f_values[i]*(1-f_values[i])

    similarity_score = pow(sum_of_residual, 2) / \
        (denominator_value + lambda_value)

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
            data=group, f_values=f_value)

    gain = similarity_score - sc_root
    return gain


def predict_output(groups, x, f_value):
    output = 0

    group_1, group_2 = groups
    denominator_value_group1 = 0
    denominator_value_group2 = 0

    for i in range(len(group_1)):
        denominator_value_group1 += f_value[i]*(1-f_value[i])

    for i in range(len(group_2)):
        denominator_value_group2 += f_value[i]*(1-f_value[i])

    for i in range(len(df['x'])):
        if x > df['x'][i]:
            if df['y'][i] in group_1:
                output = (sum(group_1) - f_value*len(group_1)) / \
                    denominator_value_group1
                break
            else:
                output = (sum(group_1) - f_value*len(group_1)) / \
                    denominator_value_group2
                break

    predicted_value = f_0 + learning_rate*output

    probaility = pow(np.e, predicted_value)/(1 + pow(np.e, predicted_value))
    print(probaility)


if __name__ == "__main__":

    sc_root = compute_similarity_score(
        data=df['y'], f_values=[0.5, 0.5, 0.5, 0.5])

    # case i
    groups_1 = split_by_threshold(data=df, threshold=23.5)
    gain_1 = compute_gain(groups_1, [0.5, 0.5, 0.5, 0.5], sc_root)

    # case ii
    groups_2 = split_by_threshold(data=df, threshold=25)
    gain_2 = compute_gain(groups_2, [0.5, 0.5, 0.5, 0.5], sc_root)

    # case iii
    groups_3 = split_by_threshold(data=df, threshold=26.5)
    gain_3 = compute_gain(groups_3, [0.5, 0.5, 0.5, 0.5], sc_root)

    max_gain = max(np.array([gain_1, gain_2, gain_3]))

    # in case of x = 25
    if max_gain == gain_1:
        predict_output(groups_1, 25, [0.5, 0.5, 0.5, 0.5])

    elif max_gain == gain_2:
        predict_output(groups_2, 25, [0.5, 0.5, 0.5, 0.5])

    else:
        predict_output(groups_3, 25, [0.5, 0.5, 0.5, 0.5])
