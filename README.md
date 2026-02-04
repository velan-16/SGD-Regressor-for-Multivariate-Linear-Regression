# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect the dataset and separate it into independent variables (house features) and dependent variables (house price and number of occupants).
2. Split the dataset into training and testing sets and apply feature scaling to the independent variables.
3. Train the multivariate linear regression model using SGD Regressor on the training data.
4. Predict the house price and number of occupants using the trained model and evaluate the results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:   SUGAVELAN S
RegisterNumber:  25005466
*/
```
```
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Size': [800, 1000, 1200, 1500, 1800],
    'Rooms': [2, 3, 3, 4, 5],
    'LocationScore': [3, 4, 5, 4, 5],
    'Price': [40, 55, 65, 80, 95],
    'Occupants': [3, 4, 5, 6, 7]
}

df = pd.DataFrame(data)

X = df[['Size', 'Rooms', 'LocationScore']]
y_price = df['Price']
y_occupants = df['Occupants']

X_train, X_test, y_price_train, y_price_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42)

_, _, y_occ_train, y_occ_test = train_test_split(
    X, y_occupants, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
price_model.fit(X_train_scaled, y_price_train)

occupant_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occupant_model.fit(X_train_scaled, y_occ_train)

price_pred = price_model.predict(X_test_scaled)
occupant_pred = occupant_model.predict(X_test_scaled)

print("House Price Prediction")
print("MSE:", mean_squared_error(y_price_test, price_pred))
print("R2 Score:", r2_score(y_price_test, price_pred))

print("\nOccupants Prediction")
print("MSE:", mean_squared_error(y_occ_test, occupant_pred))
print("R2 Score:", r2_score(y_occ_test, occupant_pred))

sample_house = np.array([[1400, 3, 4]])
sample_scaled = scaler.transform(sample_house)

print("\nPredicted House Price:", price_model.predict(sample_scaled)[0])
print("Predicted Number of Occupants:", occupant_model.predict(sample_scaled)[0])
```

## Output:
<img width="589" height="219" alt="image" src="https://github.com/user-attachments/assets/17e72ec7-44ba-4913-9159-0d871cb0ac3a" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
