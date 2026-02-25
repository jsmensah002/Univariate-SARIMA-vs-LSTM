Brief Overview:
- This project compares two univariate time series models (SARIMA and LSTM) to forecast electricity consumption, with the goal of identifying the best performing model for accurate and reliable predictions.

Method:
- The dataset was split into an 80/20 train-test ratio, with both models trained on 80% of the data and evaluated on the remaining 20%. While SARIMA provides additional diagnostic criteria such as the Ljung-Box test, Jarque-Bera test, Heteroskedasticity test and parameter significance values, LSTM does not produce equivalent statistical diagnostics. Therefore, to ensure a fair and consistent comparison between both models, Root Mean Squared Error (RMSE) was used as the primary evaluation metric for both the training and test sets. Forecast results were then visualized to compare how well each model captured the underlying trend and seasonal patterns.

SARIMA Results:
- The Auto-ARIMA selection process identified an optimal SARIMA(1, 1, 2)(0, 1, 1)[12] model after tuning. All parameters were statistically significant and the model produced the following results:
Train RMSE = 5.02
Test RMSE = 4.20

LSTM Results:
- The LSTM model was configured with 100 neurons, n_input = 12, and trained for 50 epochs. The model produced the following results:
Train RMSE = 4.12
Test RMSE = 5.57

Comparison:
- While LSTM achieved a lower train RMSE, SARIMA produced a lower test RMSE of 4.20 compared to LSTM's 5.57, indicating better generalization to unseen data. Additionally SARIMA naturally maintained the seasonal patterns in the future forecast while LSTM struggled to reproduce them, which is a known limitation of neural networks on small univariate seasonal datasets. Based on both RMSE performance and forecast quality, SARIMA is the better model for this dataset.

Graph Plotting:
- Both models were plotted showing the actual data, test predictions and future forecast. The SARIMA forecast maintained clear seasonal patterns with a confidence interval that widened naturally over time. The LSTM forecast closely followed the actual data during the test period but produced a flatter future forecast, reflecting its difficulty in reproducing seasonality without real data as input.

Key Insights:
- For datasets with clear and consistent seasonality like electricity consumption, SARIMA outperforms LSTM in both test RMSE and forecast quality. LSTM seems to be better suited for complex, irregular datasets with multiple features. Future work would explore multivariate LSTM.
- Unlike SARIMA which is a statistical model that mathematically calculates the uncertainty of its predictions, LSTM is a neural network that produces a single point prediction with no measure of uncertainty attached to it. As a result confidence intervals could not be generated for the LSTM forecast. Achieving confidence intervals with LSTM would require more advanced techniques which is beyond the scope of this project.
