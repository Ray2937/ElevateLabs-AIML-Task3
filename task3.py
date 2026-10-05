import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Import and preprocess the dataset
print('--- 1. LOADING & PREPROCESSING DATA ---')
df = pd.read_csv('Housing.csv')

# Selecting features (X) and target (y) for Multiple Linear Regression
# Let's use 'area', 'bedrooms', 'bathrooms', 'stories', and 'parking' to predict 'price'
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X = df[features]
y = df['price']

# 2. Split data into train-test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit a Linear Regression model
print('\n--- 3. FITTING LINEAR REGRESSION MODEL ---')
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 4. Evaluate model using MAE, MSE, RMSE, and R²
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Model Intercept: {model.intercept_:.2f}')
for feature, coef in zip(features, model.coef_):
    print(f'Coefficient for {feature}: {coef:.2f}')

print(f'\nMean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R² Score: {r2:.4f}')

# 5. Plot Actual vs Predicted prices and save plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.6, edgecolors='k')
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2
)
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.title('Actual vs Predicted House Prices (Linear Regression)')
plt.savefig('actual_vs_predicted_housing.png')
plt.close()

print(
    '\nTask 3 code executed successfully! Visualization saved as'
    ' actual_vs_predicted_housing.png'
)