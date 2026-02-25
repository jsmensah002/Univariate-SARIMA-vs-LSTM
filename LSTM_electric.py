import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
df = pd.read_csv('Electricity Consumption.csv')
print(df)

df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')
df.set_index('DATE',inplace = True)

df.rename(columns={'IPG2211A2N':'Consumption'},inplace=True)
print(df)
print(df.dtypes)

# All three do the same thing, fix the random starting point but for different libraries
# All three need to be set because each library has its own source of randomness, so fixing just one won't be enough to get consistent results every time.
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Train test split
size = int(len(df) * 0.8)
x_train, x_test = df[0:size], df[size:len(df)]

# Scale data
scaler = MinMaxScaler()
# scaled train would change if we have multiple columns
# eg scaled_train = scaler.fit_transform(x_train[['target column', 'column2', 'column3']])
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

# Create sequences manually
def create_sequences(data, n_input):
    x, y = [], []
    for i in range(len(data) - n_input):
        x.append(data[i: i + n_input])
        y.append(data[i + n_input])
    return np.array(x), np.array(y)

n_input = 12

# n_features would change if we have multiple columns
# eg n_features = 3
n_features = 1

x_train, y_train = create_sequences(scaled_train, n_input)
x_test, y_test = create_sequences(scaled_test, n_input)

# Define and train model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=50, verbose=0, shuffle=False)

# Train RMSE
train_predictions = model.predict(x_train)
train_predictions = scaler.inverse_transform(train_predictions)
actual_train = scaler.inverse_transform(y_train.reshape(-1, 1))
train_rmse = np.sqrt(mean_squared_error(actual_train, train_predictions))
print(f'Train RMSE: {train_rmse}')

# Test RMSE
# test_predictions would change if we have multiple columns
# test_predictions = scaler.inverse_transform(predictions)[:, 0], since the target column is [0] in line 32
test_predictions = model.predict(x_test)
test_predictions = scaler.inverse_transform(test_predictions)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
test_rmse = np.sqrt(mean_squared_error(actual, test_predictions))
print(f'Test RMSE: {test_rmse}')

# Forecast into the future
n_future = 24
last_batch = scaled_train[-n_input:].reshape(1, n_input, n_features)

future_predictions = []
for i in range(n_future):
    future_pred = model.predict(last_batch, verbose=0)
    future_predictions.append(future_pred[0])
    last_batch = np.append(last_batch[:, 1:, :], [[future_pred[0]]], axis=1)

future_predictions = scaler.inverse_transform(future_predictions)

# Plot all three: training, test predictions and future forecast in one graph
test_dates = df.index[size + n_input:]
future_dates = pd.date_range(df.index[-1], periods=n_future+1, freq='MS')[1:]

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Consumption'], label='Actual', color='red')
plt.plot(test_dates, test_predictions, label='Test Predictions', color='blue')
plt.plot(future_dates, future_predictions, label='Future Forecast', color='green')
plt.title('Electric Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.legend()
plt.show()