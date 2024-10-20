# Author: Jason Gardner
# Date: 10/20/2024
# Class: CAP6768
# Assignment: Discussion 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

FILENAME = "RentMortgage.csv"
CATEGORY = "School"

class Data:
    def __init__(self) -> None:
        self.filename = FILENAME
        self.data = self.load_data()
        self.data = self._process_data()
        
    def load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.filename)
        return data

    def _process_data(self) -> np.ndarray:
        X = self.data["Rent ($)"].values
        Y = self.data["Mortgage ($)"].values
        data = np.column_stack((X, Y))
        return data
    
    def get_data(self) -> np.ndarray:
        return self.data

class Regression:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.X = data[:, 0].reshape(-1, 1)
        self.Y = data[:, 1].reshape(-1, 1)
        self.linear_model, self.quadratic_model = self._fit_models()
        self.Y_pred_linear = self.linear_model.predict(self.X)
        self.Y_pred_quadratic = self.quadratic_model.predict(self.X)
        self.residuals = self.Y - self.Y_pred_linear
        
        # Extract coefficients for the equations
        self.linear_coef, self.linear_intercept = self._get_linear_equation()
        self.quadratic_coefs = self._get_quadratic_equation()
        
    def _fit_models(self):
        # Fit linear regression model
        linear_model = LinearRegression()
        linear_model.fit(self.X, self.Y)
        # Fit quadratic regression model
        quadratic_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        quadratic_model.fit(self.X, self.Y)
        return linear_model, quadratic_model
    
    def _get_linear_equation(self):
        # Get coefficients for linear regression equation
        coef = self.linear_model.coef_[0][0]
        intercept = self.linear_model.intercept_[0]
        return coef, intercept
    
    def _get_quadratic_equation(self):
        # Get coefficients for quadratic regression equation
        # The coefficients correspond to [intercept, x_term, x_squared_term]
        linear_reg = self.quadratic_model.named_steps['linearregression']
        coefs = linear_reg.coef_[0]
        intercept = linear_reg.intercept_[0]
        # Combine intercept and coefficients
        # Since PolynomialFeatures adds a bias term, we need to include intercept separately
        coefs[0] = intercept
        return coefs  # [c, b, a] for ax^2 + bx + c

    def get_models(self):
        return self.linear_model, self.quadratic_model

    def get_equations(self):
        return (self.linear_coef, self.linear_intercept), self.quadratic_coefs

if __name__ == "__main__":
    # Load and process data
    data_instance = Data()
    data = data_instance.get_data()
    rent = data[:, 0]
    mortgage = data[:, 1]
    
    # Part a: Scatter plot of mortgage vs. rent
    plt.figure(figsize=(10, 6))
    plt.scatter(rent, mortgage, color='blue', label='Data Points')
    plt.xlabel('Average Asking Rent ($)')
    plt.ylabel('Monthly Mortgage on Median-Priced Home ($)')
    plt.title('Scatter Plot of Mortgage vs. Rent')
    plt.legend()
    plt.savefig('scatter_plot.png')
    plt.close()
    
    # Initialize regression models
    regression = Regression(data)
    linear_model, quadratic_model = regression.get_models()
    
    # Part b: Linear regression and residual plot
    X = regression.X
    Y = regression.Y
    Y_pred_linear = regression.Y_pred_linear
    residuals = regression.residuals
    
    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(rent, residuals, color='purple')
    plt.hlines(y=0, xmin=rent.min(), xmax=rent.max(), colors='red', linestyles='--')
    plt.xlabel('Average Asking Rent ($)')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Rent')
    plt.savefig('residuals_plot.png')
    plt.close()
    
    # Part c: Quadratic regression model (already fitted)
    Y_pred_quadratic = regression.Y_pred_quadratic
    
    # Part d: Compare linear and quadratic models
    # Sort the data for plotting
    sorted_indices = np.argsort(X.flatten())
    X_sorted = X.flatten()[sorted_indices]
    Y_pred_linear_sorted = Y_pred_linear.flatten()[sorted_indices]
    Y_pred_quadratic_sorted = Y_pred_quadratic.flatten()[sorted_indices]
    
    # Plotting both regression lines over the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(rent, mortgage, color='blue', label='Data Points')
    plt.plot(X_sorted, Y_pred_linear_sorted, color='red', label='Linear Regression')
    plt.plot(X_sorted, Y_pred_quadratic_sorted, color='green', label='Quadratic Regression')
    plt.xlabel('Average Asking Rent ($)')
    plt.ylabel('Monthly Mortgage on Median-Priced Home ($)')
    plt.title('Comparison of Linear and Quadratic Regression Models')
    plt.legend()
    plt.savefig('regression_comparison.png')
    plt.close()
    
    # Extract regression equations
    (linear_coef, linear_intercept), quadratic_coefs = regression.get_equations()
    # For quadratic_coefs, the order is [intercept, coef_x, coef_x2]
    quadratic_intercept = quadratic_coefs[0]
    quadratic_coef_x = quadratic_coefs[1]
    quadratic_coef_x2 = quadratic_coefs[2]
    
    # Additional outputs (e.g., R-squared values)
    r_squared_linear = linear_model.score(X, Y)
    r_squared_quadratic = quadratic_model.score(X, Y)
    
    # Save model performance metrics and equations to a text file
    with open('model_performance.txt', 'w') as f:
        f.write('Model Performance Metrics and Equations\n')
        f.write('=======================================\n')
        f.write(f'Linear Regression Equation:\n')
        f.write(f'    Mortgage = {linear_coef:.4f} * Rent + {linear_intercept:.2f}\n')
        f.write(f'Linear Regression R-squared: {r_squared_linear:.4f}\n\n')
        f.write(f'Quadratic Regression Equation:\n')
        f.write(f'    Mortgage = {quadratic_coef_x2:.6f} * Rent^2 + {quadratic_coef_x:.4f} * Rent + {quadratic_intercept:.2f}\n')
        f.write(f'Quadratic Regression R-squared: {r_squared_quadratic:.4f}\n')
    
    # Print conclusions to a text file
    with open('conclusions.txt', 'w') as f:
        f.write('Conclusions\n')
        f.write('===========\n')
        f.write('Based on the residual plot and R-squared values, the quadratic regression model provides a better fit to the data compared to the linear regression model.\n')
        f.write('The quadratic model captures the curvature in the data, which the linear model fails to account for.\n')
        f.write('\n')
        f.write('Regression Equations:\n')
        f.write('---------------------\n')
        f.write(f'Linear Regression Equation:\n')
        f.write(f'    Mortgage = {linear_coef:.4f} * Rent + {linear_intercept:.2f}\n')
        f.write(f'Quadratic Regression Equation:\n')
        f.write(f'    Mortgage = {quadratic_coef_x2:.6f} * Rent^2 + {quadratic_coef_x:.4f} * Rent + {quadratic_intercept:.2f}\n')