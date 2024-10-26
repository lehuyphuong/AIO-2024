import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('SalesPrediction.csv')
df = pd.get_dummies(df)

# Handle null values
df = df.fillna(df.mean())

# Get features
x = df[['TV', 'Radio', 'Social Media', 'Influencer_Macro',
        'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']]

y = df[['Sales']]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=0
)

scaler = StandardScaler()
x_train_processed = scaler.fit_transform(x_train)
x_test_processed = scaler.fit_transform(x_test)
print(scaler.mean_[0])

poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(x_train_processed)
x_test_poly = poly_features.fit_transform(x_test_processed)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)

preds = poly_model.predict(x_test_poly)
print(r2_score(y_test, preds))
