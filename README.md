# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Data preparation

Step 3. Hypothesis Definition

Step 4. Cost Function

Step 5.Parameter Update Rule

Step 6.Iterative Training

Step 7.Model evaluation

Step 8.End 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Baskar.U
RegisterNumber:  212223220013
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/fee20386-8a11-4864-b68a-3b314bf11881)
```
X = df.drop(columns=['AveOccup','HousingPrice'])
X.info()
```
![image](https://github.com/user-attachments/assets/30c73843-e1b9-4efa-bfa3-7e5acee028bf)
```
Y = df[['AveOccup','HousingPrice']]
Y.info()
```
![image](https://github.com/user-attachments/assets/a64bc3aa-efa9-4344-8182-a205a6fef2d5)
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
```
![image](https://github.com/user-attachments/assets/0c52cb44-53f0-4c05-863b-f60707f6995f)
```
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
```
![image](https://github.com/user-attachments/assets/61123b39-d946-45fc-a0b2-fab72e7f6b38)

```
print("\nPredictions:\n", Y_pred[:5])
```

## Output:
![image](https://github.com/user-attachments/assets/4d040a93-87b8-44b8-86bd-9c49b706b15e)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
