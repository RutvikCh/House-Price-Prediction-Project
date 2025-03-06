# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Step 2: Load the dataset (Boston Housing Dataset from sklearn)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# Load the Boston dataset
#boston = load_boston()
X = pd.DataFrame(housing.data, columns=housing.feature_names)  # Features
y = pd.Series(housing.target)  # Target (prices)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions using the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
# 1. Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# 2. Mean Squared Error (MSE)
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 3. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# 4. R-Squared (R2) Score
r2 = metrics.r2_score(y_test, y_pred)
print(f"R-Squared Score: {r2}")

# Step 7: Visualizing the results (True vs Predicted)
plt.scatter(y_test, y_pred)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True Prices vs Predicted Prices')
plt.show()
