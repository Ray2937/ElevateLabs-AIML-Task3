# Task 3: Linear Regression (Housing Price Prediction)

## Overview
This repository contains the solution for Task 3 of the AIML Internship at Elevate Labs[cite: 5]. The objective of this task is to implement multiple linear regression using Python and Scikit-Learn to predict house prices based on various structural and feature attributes[cite: 5].

## Files Included
- `task3.py`: The complete Python script for data preparation, model training, evaluation, and visualization.
- `Housing.csv`: The dataset used for regression modeling.
- `actual_vs_predicted_housing.png`: Scatter plot visualization comparing actual house prices against model predictions.

## Workflow & Steps Performed
1. **Data Loading & Selection:** Loaded the housing dataset and selected key numerical predictor variables (`area`, `bedrooms`, `bathrooms`, `stories`, and `parking`) to predict the target variable (`price`).
2. **Train-Test Split:** Split the dataset into 80% training data and 20% testing data using Scikit-Learn's `train_test_split`.
3. **Model Training:** Fitted a Multiple Linear Regression model (`LinearRegression`) on the training subset.
4. **Model Evaluation:** Evaluated model performance using standard regression metrics:
   - **Mean Absolute Error (MAE)**
   - **Mean Squared Error (MSE)**
   - **Root Mean Squared Error (RMSE)**
   - **$R^2$ Score (Coefficient of Determination)**
5. **Visualization:** Generated an actual vs. predicted price plot with a 45-degree reference line to visually assess model accuracy.