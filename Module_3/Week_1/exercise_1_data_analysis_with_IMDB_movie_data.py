import numpy as np
import pandas as pd
import matplotlib as plt

dataset_path = 'IMDB-Movie-Data.csv'
REVENUE_MILLION = 'Revenue (Millions)'

print("--------------Section 1: Read data from .csv file ------------------")
# Section 1: Read data from .csv file
data = pd.read_csv(dataset_path)
data_indexed = pd.read_csv(dataset_path, index_col='Title')


print("------------------------Section 2: View data -----------------------")
# Section 2: View data
print(data.head())


print("----------Section 3: Understand some basic information -------------")
# Section 3: Understand some basic information about data
data.info()


print("----------------Section 4: Index and Slice data --------------------")
# Section 4: Index and Slice data (Data selection)
genre = data['Genre']
print(genre)

some_cols = data[['Title', 'Genre', 'Actors', 'Director', 'Rating']]
print(some_cols)

print(data.iloc[10: 15][['Title', 'Rating', REVENUE_MILLION]])


print("------Section 5: Select data based on conditional filtering --------")
# Section 5: Select data based on conditional filtering(Data selection)
print(data[((data['Year'] >= 2010) & (data['Year'] <= 2015))
           & (data['Rating'] < 6.0)
           & (data[REVENUE_MILLION] > data[REVENUE_MILLION]
              .quantile(0.95))])


print("--------------Section 6: Group data -----------------")
# Section 6: Group data
print(data.groupby('Director')[['Rating']].mean().head())


print("--------------------------Section 7: Sort data ---------------------")
# Section 7: Sort data
print(data.groupby('Director')[['Rating']].mean(
).sort_values(['Rating'], ascending=False).head())


print("--------------Section 8: View missing data -------------------------")
# Section 8: View missing data
print(data.isnull().sum())


print("--------------Section 9: Deal with missing data (Deleting)----------")
# Section 9: Deal with missing data - deleting
print(data.dropna())


print("-------Section 10: Deal with missing data (Filling)-----------------")
# Section 10: Deal with missing data - filling
revenue_mean = data_indexed[REVENUE_MILLION].mean()
print("The mean revenue is: ", revenue_mean)

data_indexed[REVENUE_MILLION].fillna({REVENUE_MILLION:
                                      data_indexed[REVENUE_MILLION]
                                      .mean()}, inplace=True)


print("--------------Section 11: Apply function -----------------")
# Section 11: Apply function


def rating_group(rating):
    if rating >= 7.5:
        return 'Good'
    elif rating >= 6.0:
        return 'Average'
    else:
        return 'Bad'


data['Rating_category'] = data['Rating'].apply(rating_group)
print(data[['Title', 'Director', 'Rating', 'Rating_category']].head(5))
