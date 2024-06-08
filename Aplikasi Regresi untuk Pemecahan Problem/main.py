import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('Student_Performance.csv')

# Extracting relevant columns based on the problem statement
# Problem 2: Number of Practice Questions (NL) vs Test Scores (NT)
X_problem2 = df[['Sample Question Papers Practiced']].values
y = df['Performance Index'].values

# Method 1: Linear Regression
linear_model2 = LinearRegression()
linear_model2.fit(X_problem2, y)
y_pred_linear2 = linear_model2.predict(X_problem2)

# Method 2: Simple Power Regression
# log transform both X and y for a linear fit
X_problem2_log = np.log(X_problem2 + 1)  # Adding 1 to avoid log(0)
y_log = np.log(y + 1)  # Adding 1 to avoid log(0)
power_model2 = LinearRegression()
power_model2.fit(X_problem2_log, y_log)
y_pred_log_power2 = power_model2.predict(X_problem2_log)
y_pred_power2 = np.exp(y_pred_log_power2) - 1  # Transform back

# Plotting
plt.figure(figsize=(10, 5))

# Problem 2: Number of Practice Questions (NL) vs Test Scores (NT)
plt.scatter(X_problem2, y, color='blue', label='Data Points')
plt.plot(X_problem2, y_pred_linear2, color='red', label='Linear Regression')
plt.plot(X_problem2, y_pred_power2, color='green', label='Power Regression')
plt.xlabel('Sample Question Papers Practiced')
plt.ylabel('Performance Index')
plt.title('Problem 2: Number of Practice Questions vs Test Scores')
plt.legend()
plt.show()

# Model Evaluation
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rms = np.sqrt(mse)
    print(f"Model: {model_name}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rms}")
    print()

# Evaluate Linear Regression
evaluate_model(linear_model2, X_problem2, y, "Linear Regression (Problem 2)")

# Evaluate Power Regression
mse_power2 = mean_squared_error(y, y_pred_power2)
rms_power2 = np.sqrt(mse_power2)
print(f"Model: Power Regression (Problem 2)")
print(f"Mean Squared Error: {mse_power2}")
print(f"Root Mean Squared Error: {rms_power2}")
print()
