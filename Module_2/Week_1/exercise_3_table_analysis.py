import pandas as pd
import numpy as np

df = pd.read_csv(
    "D:/2024/AIO2024/GIT/AIO-2024/Module_2/Week_1/advertising.csv")
data = df.to_numpy()

tv_data = np.zeros(shape=(len(data[:, 0])))
radio_data = np.zeros(shape=(len(data[:, 0])))
newspaper_data = np.zeros(shape=(len(data[:, 0])))
sale_data = np.zeros(shape=(len(data[:, 0])))


if __name__ == "__main__":

    for i in range(len(data[:, 0])):
        tv_data[i] = data[i, 0]
        radio_data[i] = data[i, 1]
        newspaper_data[i] = data[i, 2]
        sale_data[i] = data[i, 3]

    # Exercise 15: Locate maximum number of sale
    max_sale_value = np.max(sale_data)
    index_max_sale_value = np.unravel_index(
        np.argmax(sale_data), sale_data.shape)
    print([max_sale_value, index_max_sale_value])
    # -> Answer: C

    # Exercise 16: Calculate average TV
    average_tv_value = np.average(tv_data)
    print(average_tv_value)
    # -> Answer: B

    # Exercise 17: Calculate number of sales that are greater than 20
    number_of_sale_data_greater_20 = np.count_nonzero(
        np.where(sale_data >= 20, True, False))
    print(number_of_sale_data_greater_20)
    # -> Answer: A

    # Exercise 18: Calculate average of Radio whose sale are greater than 15
    average_radio_value = 0
    sale_data_greater_15 = np.where(sale_data >= 15, True, False)
    number_of_sale_greater_15 = np.count_nonzero(sale_data_greater_15)
    for i in range(len(radio_data)):
        if sale_data_greater_15[i] == True:
            average_radio_value += radio_data[i]/number_of_sale_greater_15
    print(average_radio_value)
    # -> Answer: B

    # Exercise 19: Calculate number of sales whose news are greater than avg
    Average_newspaper = np.average(newspaper_data)
    sum_of_sale_greater_avg_news = np.sum(
        np.where(newspaper_data > Average_newspaper, sale_data, 0))
    print(sum_of_sale_greater_avg_news)
    # -> Answer: C

    # Exercise 20: Print score[7:10]
    A = np.average(sale_data)
    scores = ['Good' if x > A else 'Bad' if x <
              A else 'Average' for x in sale_data]
    print(scores[7:10])
    # -> Answer: C

    # Exercise 21: Print score[7:10]
    A = np.average(sale_data)
    scores = ['Good' if x > A else 'Bad' if x <
              A else 'Average' for x in df['Sales']]
    print(scores[7:10])
    # -> Answer: C
