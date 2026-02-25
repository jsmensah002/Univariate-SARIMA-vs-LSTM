import pandas as pd
df = pd.read_csv('Electricity Consumption.csv')
print(df)

df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')
df.set_index('DATE',inplace = True)

df.rename(columns={'IPG2211A2N':'Consumption'},inplace=True)
print(df)
print(df.dtypes)

# testing whether a data is is stationary or not using p values
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df['Consumption'])
print('Pvalue = ', pvalue)

from statsmodels.tsa.statespace.sarimax import SARIMAX
#train test split
size = int(len(df) * 0.8)
x_train, x_test = df[0:size], df[size:len(df)]

from pmdarima import auto_arima
arima_model = auto_arima(x_train['Consumption'],start_p = 1,
                         d = 1,
                         start_q = 1,
                         max_p = 5,max_q = 5, max_d = 5,m = 12, D = 1,
                         start_P = 0, start_Q = 0, max_P = 5,
                         max_D = 5,max_Q = 5,
                         seasonal = True,trace = False,
                         error_action = 'ignore', suppress_warnings = True,
                         stepwise = True, n_fits = 50)

print(arima_model.summary())

model = SARIMAX(x_train['Consumption'],
                order = (1, 1, 2),
                seasonal_order = (0, 1, 1, 12))
result = model.fit()
print(result.summary()) 

#train prediction
start_index = 0
end_index = len(x_train)-1
train_prediction = result.predict(start_index, end_index)

#test prediction of the next few months
start_index = len(x_train)
end_index = len(df)-1
test_prediction = result.predict(start_index, end_index)

import math
from sklearn.metrics import mean_squared_error

#calculate root mean squared error
train_score = math.sqrt(mean_squared_error(x_train['Consumption'],train_prediction))
print('Train Score RMSE = ', train_score)

test_score = math.sqrt(mean_squared_error(x_test['Consumption'],test_prediction))
print('Test Score RMSE = ',test_score)

#Plotting forecasting graph with confidence interval
forecast_results = result.get_forecast(steps=len(x_test) + 5*12)
forecast_mean = forecast_results.predicted_mean
confidence_intervals = forecast_results.conf_int()

print(forecast_mean)

import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.plot(x_train['Consumption'], label='Training', color='red')
plt.plot(forecast_mean, label='Forecast', color='green')

# shading between lower and upper bound
plt.fill_between(confidence_intervals.index,
                 confidence_intervals.iloc[:,0],
                 confidence_intervals.iloc[:,1],
                 color='green', alpha=0.2, label='Confidence Interval')

# plotting test AFTER fill_between so it appears on top
plt.plot(x_test['Consumption'], label='Test', color='yellow')

plt.title('SARIMA Electricity Consumption Forecast')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.ylim(bottom=0)
plt.show()
