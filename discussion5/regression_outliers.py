import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import zscore
import statsmodels.api as sm

FILENAME = "RentMortgage.csv"

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
        self.linear_model, self.quadratic_model, self.degree_9_model, self.degree_10_model, self.degree_20_model = self._fit_models()
        self.Y_pred_linear = self.linear_model.predict(self.X)
        self.Y_pred_quadratic = self.quadratic_model.predict(self.X)
        self.Y_pred_degree_9 = self.degree_9_model.predict(self.X)
        self.Y_pred_degree_10 = self.degree_10_model.predict(self.X)
        self.Y_pred_degree_20 = self.degree_20_model.predict(self.X)
        self.residuals = self.Y - self.Y_pred_linear
        
        # Extract coefficients for the equations
        self.linear_coef, self.linear_intercept = self._get_linear_equation()
        self.quadratic_intercept, self.quadratic_coef_x, self.quadratic_coef_x2 = self._get_quadratic_equation()
        self.degree_9_coefs = self._get_poly_coefficients(self.degree_9_model)
        self.degree_10_coefs = self._get_poly_coefficients(self.degree_10_model)
        self.degree_20_coefs = self._get_poly_coefficients(self.degree_20_model)
    
    def _fit_models(self):
        # Fit linear regression model
        linear_model = LinearRegression()
        linear_model.fit(self.X, self.Y)
        
        # Fit quadratic regression model
        quadratic_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        quadratic_model.fit(self.X, self.Y)
        
        # Fit 8th-degree polynomial regression model
        degree_9_model = make_pipeline(PolynomialFeatures(9), LinearRegression())
        degree_9_model.fit(self.X, self.Y)
        
        # Fit 10th-degree polynomial regression model
        degree_10_model = make_pipeline(PolynomialFeatures(10), LinearRegression())
        degree_10_model.fit(self.X, self.Y)
        
        # Fit 20th-degree polynomial regression model
        degree_20_model = make_pipeline(PolynomialFeatures(20), LinearRegression())
        degree_20_model.fit(self.X, self.Y)
        
        return linear_model, quadratic_model, degree_9_model, degree_10_model, degree_20_model
    
    def _get_linear_equation(self):
        # Get coefficients for linear regression equation
        coef = self.linear_model.coef_[0][0]
        intercept = self.linear_model.intercept_[0]
        return coef, intercept
    
    def _get_quadratic_equation(self):
        # Get coefficients for quadratic regression equation
        linear_reg = self.quadratic_model.named_steps['linearregression']
        coefs = linear_reg.coef_[0]
        intercept = linear_reg.intercept_[0]
        return intercept, coefs[1], coefs[2]  # Return intercept, coef_x, coef_x2
    
    def _get_poly_coefficients(self, model):
        # Get coefficients for a polynomial regression model
        linear_reg = model.named_steps['linearregression']
        return linear_reg.intercept_[0], linear_reg.coef_[0]  # Return intercept and array of coefficients
    
    def get_models(self):
        return self.linear_model, self.quadratic_model, self.degree_9_model, self.degree_10_model, self.degree_20_model

    def get_equations(self):
        return (
            (self.linear_coef, self.linear_intercept),
            (self.quadratic_intercept, self.quadratic_coef_x, self.quadratic_coef_x2),
            self.degree_9_coefs,
            self.degree_10_coefs,
            self.degree_20_coefs
        )
    
    def detect_outliers_zscore(self):
        z_scores = zscore(self.Y)  # Z-scores for the Y (Mortgage) values only
        return np.where(np.abs(z_scores) > 3)[0]  # Indices of outliers (Z-score > 3 or < -3)
    
    def detect_outliers_iqr(self):
        # Corrected IQR calculation for Mortgage (dependent variable)
        Q1 = np.percentile(self.Y, 25)
        Q3 = np.percentile(self.Y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Detect indices of Y values that fall outside the IQR bounds
        return np.where((self.Y < lower_bound) | (self.Y > upper_bound))[0]
    
    def detect_outliers_cooks_distance(self):
        # Cook's distance on a linear regression model
        X_with_const = sm.add_constant(self.X)  # Adding constant for intercept
        model = sm.OLS(self.Y, X_with_const).fit()
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / len(self.Y)  # Common threshold for Cook's distance
        return np.where(cooks_d > threshold)[0]
    
    def remove_outliers(self, indices):
        # Remove rows with outlier indices
        return np.delete(self.data, indices, axis=0)

if __name__ == "__main__":
    # Load and process original data
    data_instance = Data()
    original_data = data_instance.get_data()
    
    # Initialize regression models with original data
    regression = Regression(original_data)
    
    # Detect outliers using different methods
    zscore_outliers = regression.detect_outliers_zscore()
    iqr_outliers = regression.detect_outliers_iqr()
    cooks_outliers = regression.detect_outliers_cooks_distance()
    
    # Combine outlier indices from all methods and remove duplicates
    all_outliers = set(zscore_outliers).union(iqr_outliers).union(cooks_outliers)
    
    # Print actual data points identified as outliers for verification
    print("Data points identified as outliers (before removal):")
    for idx in sorted(all_outliers):
        print(f"Index: {idx}, Rent: {original_data[idx, 0]}, Mortgage: {original_data[idx, 1]}")
    
    # Remove outliers and print cleaned data for verification
    cleaned_data = regression.remove_outliers(list(all_outliers))
    
    # Refit models with cleaned data
    cleaned_regression = Regression(cleaned_data)
    
    rent = cleaned_data[:, 0]
    mortgage = cleaned_data[:, 1]
    
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
    linear_model, quadratic_model, degree_9_model, degree_10_model, degree_20_model = cleaned_regression.get_models()
    
    # Part b: Linear regression and residual plot
    X = cleaned_regression.X
    Y = cleaned_regression.Y
    Y_pred_linear = cleaned_regression.Y_pred_linear
    residuals = cleaned_regression.residuals
    
    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(rent, residuals, color='purple')
    plt.hlines(y=0, xmin=rent.min(), xmax=rent.max(), colors='red', linestyles='--')
    plt.xlabel('Average Asking Rent ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals vs. Rent')
    plt.savefig('residuals_plot.png')
    plt.close()
    
    # Part c: Quadratic regression model (already fitted)
    Y_pred_quadratic = cleaned_regression.Y_pred_quadratic
    
    # Part d: Compare linear, quadratic, and higher-degree models
    sorted_indices = np.argsort(X.flatten())
    X_sorted = X.flatten()[sorted_indices]
    Y_pred_linear_sorted = Y_pred_linear.flatten()[sorted_indices]
    Y_pred_quadratic_sorted = Y_pred_quadratic.flatten()[sorted_indices]
    Y_pred_degree_9_sorted = cleaned_regression.Y_pred_degree_9.flatten()[sorted_indices]
    Y_pred_degree_10_sorted = cleaned_regression.Y_pred_degree_10.flatten()[sorted_indices]
    Y_pred_degree_20_sorted = cleaned_regression.Y_pred_degree_20.flatten()[sorted_indices]
    
    # Plot all regression models over the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(rent, mortgage, color='blue', label='Data Points')
    plt.plot(X_sorted, Y_pred_linear_sorted, color='red', label='Linear Regression')
    plt.plot(X_sorted, Y_pred_quadratic_sorted, color='green', label='Quadratic Regression')
    plt.plot(X_sorted, Y_pred_degree_9_sorted, color='orange', label='Degree-9 Polynomial Regression')
    plt.plot(X_sorted, Y_pred_degree_10_sorted, color='purple', label='Degree-10 Polynomial Regression')
    plt.plot(X_sorted, Y_pred_degree_20_sorted, color='brown', label='Degree-20 Polynomial Regression')
    plt.xlabel('Average Asking Rent ($)')
    plt.ylabel('Monthly Mortgage on Median-Priced Home ($)')
    plt.title('Comparison of Linear, Quadratic, Degree-9, Degree-10, and Degree-20 Polynomial Regression Models')
    plt.legend()
    plt.savefig('regression_comparison_all.png')
    plt.close()
    
    # Extract regression equations
    (linear_coef, linear_intercept), (quad_intercept, quad_coef_x, quad_coef_x2), degree_9_coefs, degree_10_coefs, degree_20_coefs = cleaned_regression.get_equations()
    
    # Additional outputs (e.g., R-squared values)
    r_squared_linear = linear_model.score(X, Y)
    r_squared_quadratic = quadratic_model.score(X, Y)
    r_squared_degree_9 = degree_9_model.score(X, Y)
    r_squared_degree_10 = degree_10_model.score(X, Y)
    r_squared_degree_20 = degree_20_model.score(X, Y)
    
    # Save model performance metrics and equations to a text file
    with open('model_performance.txt', 'w') as f:
        f.write('Model Performance Metrics and Equations\n')
        f.write('=======================================\n')
        f.write(f'Linear Regression Equation:\n')
        f.write(f'    Mortgage = {linear_coef:.4f} * Rent + {linear_intercept:.2f}\n')
        f.write(f'Linear Regression R-squared: {r_squared_linear:.4f}\n\n')
        
        f.write(f'Quadratic Regression Equation:\n')
        f.write(f'    Mortgage = {quad_intercept:.2f} + {quad_coef_x:.4f} * Rent + {quad_coef_x2:.6f} * Rent^2\n')
        f.write(f'Quadratic Regression R-squared: {r_squared_quadratic:.4f}\n\n')
        
        f.write(f'Degree-9 Polynomial Regression Coefficients:\n')
        f.write(f'    Intercept: {degree_9_coefs[0]:.4f}\n')
        f.write(f'    Coefficients: {degree_9_coefs[1][1:]}\n')  # Skipping bias term
        f.write(f'Degree-9 Polynomial Regression R-squared: {r_squared_degree_9:.4f}\n\n')
        
        f.write(f'Degree-10 Polynomial Regression Coefficients:\n')
        f.write(f'    Intercept: {degree_10_coefs[0]:.4f}\n')
        f.write(f'    Coefficients: {degree_10_coefs[1][1:]}\n')  # Skipping bias term
        f.write(f'Degree-10 Polynomial Regression R-squared: {r_squared_degree_10:.4f}\n\n')
        
        f.write(f'Degree-20 Polynomial Regression Coefficients:\n')
        f.write(f'    Intercept: {degree_20_coefs[0]:.4f}\n')
        f.write(f'    Coefficients: {degree_20_coefs[1][1:]}\n')  # Skipping bias term
        f.write(f'Degree-20 Polynomial Regression R-squared: {r_squared_degree_20:.4f}\n')
