import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv(
    "D:/2024/AIO2024/GIT/AIO-2024/Module_2/Week_4/advertising.csv")


def correlation(x, y):
    assert len(x) == len(y)
    N = len(x)
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
        sum_of_product += x[i]*y[i]
        sum_of_x += x[i]
        sum_of_y += y[i]

    # total x_i^2, total y_i^2
    for i in range(N):
        product_of_squared_x += x[i]**2
        product_of_squared_y += y[i]**2

    squared_of_product_x = sum_of_x**2
    squared_of_product_y = sum_of_y**2

    numerator = N*sum_of_product - sum_of_x * sum_of_y
    denominator = ((N*product_of_squared_x - squared_of_product_x)
                   * (N*product_of_squared_y - squared_of_product_y))**0.5
    return np.round(numerator / denominator, 2)


if __name__ == "__main__":
    x = data['TV']
    y = data['Radio']

    corr_xy = correlation(x, y)

    print(f"Correlation between TV and sale: {round(corr_xy, 2)}")
    print("--------------------------------------")

    features = ['TV', 'Radio', 'Newspaper']

    for feature_1 in features:
        for feature_2 in features:
            correlation_value = correlation(data[feature_1], data[feature_2])
            print(f" Correlation between {feature_1} and {feature_2}: {round(
                correlation_value, 2)}")

    print("--------------------------------------")
    x = data['Radio']
    y = data['Newspaper']
    result = np.corrcoef(x, y)
    print(result)

    print("--------------------------------------")
    print(data.corr(method='pearson'))

    print("--------------------------------------")
    sns.heatmap(data.corr(method='pearson'),
                annot=True, fmt=".2f", linewidths=.5,)
