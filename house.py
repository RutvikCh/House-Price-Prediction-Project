

import pandas as pd

from sklearn.datasets import fetch_california_housing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)

print(df.head())

df['MedHouseVal'] = housing.target

X = df.drop('MedHouseVal', axis=1)
y=df['MedHouseVal']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

regression = LinearRegression()
regression.fit(X_train,y_train)

mse = cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)

np.mean(mse)

reg_pred = regression.predict(X_test)

reg_pred

import seaborn as sns
sns.kdeplot(reg_pred - y_test)
plt.title("Residuals Distribution")
plt.xlabel("Prediction Error")
plt.show()

from sklearn.metrics import r2_score

score = r2_score(y_test, reg_pred)

score


# Ridge regression


from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

ridge_regressor = Ridge()

ridge_regressor

parameters = {'alpha':[1,2,5,10,20,30,40,50,60,70,80,90]}
ridgecv = GridSearchCV(ridge_regressor,parameters,scoring='neg_mean_squared_error',cv=5)
ridgecv.fit(X_train,y_train)

print(ridgecv.best_params_)

print(ridgecv.best_score_)

ridge_pred = ridgecv.predict(X_test)

import seaborn as sns
sns.kdeplot(ridge_pred - y_test)
plt.title("Residuals Distribution")
plt.xlabel("Prediction Error")
plt.show()

from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)
score



##Lasso Regression


from sklearn.linear_model import Lasso

Lasso = Lasso()

parameters = {'alpha':[1,2,5,10,20,30,40,50,60,70,80,90]}
Lassocv = GridSearchCV(Lasso,parameters,scoring='neg_mean_squared_error',cv=5)
Lassocv.fit(X_train,y_train)

print(Lassocv.best_params_)
print(Lassocv.best_score_)

Lasso_pred = Lassocv.predict(X_test)

import seaborn as sns
sns.kdeplot(Lasso_pred - y_test)
plt.title("Residuals Distribution")
plt.xlabel("Prediction Error")
plt.show()

from sklearn.metrics import r2_score
score = r2_score(y_test, reg_pred)
score