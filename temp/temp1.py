import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the stock data
df = pd.read_csv('/home/kingas/Desktop/stock_market/upload_files/hcltech_5y.csv')

# Extract the relevant features
X = df[['Open', 'High', 'Low', 'Volume']].values
y = df['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)
print(y_pred[0])

# Evaluate the model
score = regressor.score(X_test, y_test)
print('R2 Score:', score)

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('Price Predictions')
plt.show()
