import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the stock data
df = pd.read_csv(
    '/home/kingas/Desktop/stock_market/upload_files/hcltech_5y.csv')

# Extract the relevant features
X = df[['Open', 'High', 'Low', 'Volume']].values
y = df['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)
print(format(int(y_pred[0])))
# Evaluate the linear regression model
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE:', lin_rmse)

# print("hi")
# # Train a support vector machine model
# svr_regressor = SVR(kernel='linear')
# svr_regressor.fit(X_train, y_train)
# print("hi1")
# # Make predictions on the test set
# y_pred = svr_regressor.predict(X_test)
# print("hi3")
# # Evaluate the support vector machine model
# svr_mse = mean_squared_error(y_test, y_pred)
# svr_rmse = np.sqrt(svr_mse)
# print('Support Vector Machine RMSE:', svr_rmse)

# Train a decision tree model
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree_regressor.predict(X_test)
print(format(int(y_pred[0])))

# Evaluate the decision tree model
tree_mse = mean_squared_error(y_test, y_pred)
tree_rmse = np.sqrt(tree_mse)
print('Decision Tree RMSE:', tree_rmse)

# Train a random forest model
forest_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
forest_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = forest_regressor.predict(X_test)
print(format(int(y_pred[0])))


# Evaluate the random forest model
forest_mse = mean_squared_error(y_test, y_pred)
forest_rmse = np.sqrt(forest_mse)
print('Random Forest RMSE:', forest_rmse)

