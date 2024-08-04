import numpy as np


def compute_mean(array):

    return np.average(array)


def compute_median(array):
    size = len(array)
    array = np.sort(array)

    print(array)

    if (size % 2 == 0):
        return np.sum([array[int(size/2) - 1], array[int(size/2 + 1) - 1]]) * 0.5
    else:
        return array[int(size+1) / 2]


def compute_std(array):
    mean = compute_mean(array=array)
    variance = 0
    dff = []
    for number in array:
        dff.append(number - mean)

    squared_diff = []
    for number in dff:
        squared_diff.append(number**2)

    variance = np.sum(squared_diff)/len(array)
    return np.sqrt(variance)


def compute_correlation_cofficient(array_1, array_2):
    assert len(array_1) == len(array_2)
    N = len(array_1)
    numerator = 0
    denominator = 0

    product_of_squared_x = 0
    product_of_squared_y = 0

    squared_of_product_x = 0
    squared_of_product_y = 0

    sum_of_x = 0
    sum_of_y = 0

    sum_of_product = 0

    # x_i * y_i, total x_i, total y_i
    for i in range(N):
        sum_of_product += array_1[i]*array_2[i]
        sum_of_x += array_1[i]
        sum_of_y += array_2[i]

    # total x_i^2, total y_i^2
    for i in range(N):
        product_of_squared_x += array_1[i]**2
        product_of_squared_y += array_2[i]**2

    squared_of_product_x = sum_of_x**2
    squared_of_product_y = sum_of_y**2

    numerator = N*sum_of_product - sum_of_x * sum_of_y
    denominator = ((N*product_of_squared_x - squared_of_product_x)
                   * (N*product_of_squared_y - squared_of_product_y))**0.5
    return np.round(numerator / denominator, 2)


if __name__ == "__main__":
    X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
    print("Mean: ", compute_mean(X))

    X = [1, 5, 4, 4, 9, 13]
    print("Median: ", compute_median(X))

    X = [171, 176, 155, 167, 169, 182]
    print("variance: ", compute_std(X))

    X = [-2, -5, -11, 6, 4, 15, 9]
    Y = [4, 25, 121, 36, 16, 225, 81]
    print("Correlation : ", compute_correlation_cofficient(X, Y))
