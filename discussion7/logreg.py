#!/bin/python3

# Author: Jason Gardner (n01480000)
# Date: 11/11/2024
# Class: CAP6768
# Assignment: Discussion 7

'''
Salmons Stores operates a national chain of women’s apparel stores. Five thousand copies of an expensive 
four-color sales catalog have been printed, and each catalog includes a coupon that provides a $50 
discount on purchases of $200 or more. Salmons would like to send the catalogs only to customers who 
have the highest probability of using the coupon. For each of 1,000 Salmons customers, three variables 
were tracked from an earlier promotional campaign: last year’s total spending at Salmons (Spending), 
whether they have a Salmons store credit card (Card), and whether they used the promotional coupon 
they were sent (Coupon). Apply logistic regression to classify observations as a promotion-responder 
or not by using Spending and Card as input variables and Coupon as the target (or response) variable.

Salmons.xlsx

Download Salmons.xlsx 

a. Evaluate candidate logistic regression models based on their predictive performance on the validation set. 
Recommend a final model and express the model as a mathematical equation relating the target variable to 
the input variables.
b. For the model selected in part (a), provide and interpret the lift measure on the top 10% of the 
test set observations most likely to use the promotional coupon.
c. What is the area under the ROC curve on the test set? To achieve a sensitivity of at least 0.80, how 
much Class 0 error rate must be tolerated?
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import joblib
import seaborn as sns
import os, sys, warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

FILENAME = "Salmons.csv"

class Data:
    def __init__(self) -> None:
        self.filename = FILENAME
        self.data = self.load_data()
        self.data = self._process_data()

    def load_data(self) -> pd.DataFrame:
        # Load the CSV data
        data = pd.read_csv(self.filename)
        return data

    def _process_data(self) -> pd.DataFrame:
        # Processing data, selecting necessary columns
        self.data = self.data[['Spending', 'Card', 'Coupon', 'Partition']]
        return self.data

    def get_data(self) -> pd.DataFrame:
        return self.data

class LogReg:
    def __init__(self, data: pd.DataFrame) -> None:
        # Splitting into input and target variables
        self.data = data
        self.X = data[['Spending', 'Card']].values
        self.Y = data['Coupon'].values

        # Splitting into training, validation, and test sets based on 'Partition'
        train_data = data[data['Partition'] == 't']
        val_data = data[data['Partition'] == 'v']
        test_data = data[data['Partition'] == 's']

        self.X_train, self.Y_train = train_data[['Spending', 'Card']], train_data['Coupon']
        self.X_val, self.Y_val = val_data[['Spending', 'Card']], val_data['Coupon']
        self.X_test, self.Y_test = test_data[['Spending', 'Card']], test_data['Coupon']

        # Fit multiple logistic regression models with different hyperparameters
        self.models = self._fit_models()
        self.best_model, self.best_solver, self.best_c, self.best_auc = self._select_best_model()

    def _fit_models(self):
        # Initialize and fit multiple logistic regression models with different solvers and regularization
        models = []
        solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
        c_values = [0.01, 0.1, 1, 10, 100]
        
        for solver in solvers:
            for c in c_values:
                model = LogisticRegression(C=c, solver=solver)
                model.fit(self.X_train, self.Y_train)
                models.append((model, solver, c))
        return models

    def _select_best_model(self):
        # Evaluate all models on the validation set and select the best one based on AUC
        best_auc = 0
        best_model = None
        for model, solver, c in self.models:
            val_predictions = model.predict_proba(self.X_val)[:, 1]
            auc_score = roc_auc_score(self.Y_val, val_predictions)
            print(f"Model (solver={solver}, C={c}):\nValidation AUC = {auc_score}")
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model
                best_solver = solver
                best_c = c
        return best_model, best_solver, best_c, best_auc
        return best_model

    def evaluate_model(self):
        # Evaluate the selected model on the test set
        test_predictions = self.best_model.predict_proba(self.X_test)[:, 1]
        auc_score_test = roc_auc_score(self.Y_test, test_predictions)
        print(f"Test AUC: {auc_score_test}")
        with open("test_auc_description.txt", "w") as f:
            f.write(f"Test AUC: {auc_score_test}\nThis AUC indicates the overall performance of the selected model on the test dataset, assessing its classification capability.")
        with open("test_auc_score.txt", "w") as f:
            f.write(f"Test AUC: {auc_score_test}")
        
        # Save the AUC score to a file
        with open("test_auc_score.txt", "w") as f:
            f.write(f"Test AUC: {auc_score_test}")        
        return auc_score_test

    def lift_measure(self):
        # Calculate lift for top 10% of test set observations
        test_predictions = self.best_model.predict_proba(self.X_test)[:, 1]
        test_data_sorted = pd.DataFrame({'Predicted_Probability': test_predictions, 'Coupon': self.Y_test})\
                                    .sort_values(by='Predicted_Probability', ascending=False)
        top_10_percent = test_data_sorted.iloc[:int(len(test_data_sorted) * 0.1)]
        lift = top_10_percent['Coupon'].mean() / self.Y_test.mean()
        print(f"Lift in top 10%: {lift}")
        
        # Save the lift measure to a file
        with open("lift_measure.txt", "w") as f:
            f.write(f"Lift in top 10%: {lift}")        
        return lift

    def plot_roc_curve(self):
        # Plot ROC curve
        test_predictions = self.best_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.Y_test, test_predictions)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        plt.grid(True)
        plt.savefig("roc_curve.png")
        
    def sensitivity_vs_specificity(self, sensitivity_target=0.80):
        # Calculate sensitivity and required specificity
        test_predictions = self.best_model.predict(self.X_test)
        tn, fp, fn, tp = confusion_matrix(self.Y_test, test_predictions).ravel()

        # Sensitivity calculation
        sensitivity = tp / (tp + fn)
        print(f"Sensitivity: {sensitivity}")

        # Specificity
        specificity = tn / (tn + fp)
        print(f"Specificity: {specificity}")

        # Determine Class 0 error rate tolerated to achieve the desired sensitivity
        class_0_error = fp / (tn + fp)
        print(f"Class 0 Error Rate: {class_0_error}")

        # Save sensitivity and class 0 error rate to a file
        with open("sensitivity_specificity.txt", "w") as f:
            f.write(f"Sensitivity: {sensitivity}")
            f.write(f"\nSpecificity: {specificity}")
            f.write(f"\nClass 0 Error Rate: {class_0_error}")
            f.write(f"To achieve a sensitivity of at least 0.80, the model may need to tolerate a higher Class 0 error rate, as calculated above.")        
        return sensitivity, class_0_error

    def output_confusion_matrix(self):
        # Output the confusion matrix for the best model on the test set
        test_predictions = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.Y_test, test_predictions)
        class_names = ['No Coupon', 'Coupon']

        # Plot confusion matrix using seaborn heatmap for better visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        
        # Save confusion matrix to a file (numeric values)
        with open("confusion_matrix.txt", "w") as f:
            f.write(f"Confusion Matrix: {cm}")
            
if __name__ == "__main__":
    # Load and process data
    data_instance = Data()
    data = data_instance.get_data()

    # Train and evaluate logistic regression model
    log_reg = LogReg(data)
    auc_score = log_reg.evaluate_model()
    lift = log_reg.lift_measure()
    log_reg.plot_roc_curve()
    sensitivity, class_0_error = log_reg.sensitivity_vs_specificity()
    log_reg.output_confusion_matrix()

    # Save the best model to a file
    joblib.dump(log_reg.best_model, "best_logistic_regression_model.pkl")

    # Output the final model equation
    coef = log_reg.best_model.coef_[0]
    intercept = log_reg.best_model.intercept_[0]
    equation = f"Coupon = 1 / (1 + exp(-({intercept} + {coef[0]} * Spending + {coef[1]} * Card)))"
    print(f"Final Model Equation: {equation}")
    print(f"Best Model Description: Solver={log_reg.best_solver}, C={log_reg.best_c}, AUC={log_reg.best_auc}")
    with open("best_model_description.txt", "w") as f:
        f.write(f"Best Model Description: Solver={log_reg.best_solver}, C={log_reg.best_c}, AUC={log_reg.best_auc}\nThe final model uses the solver '{log_reg.best_solver}' with regularization parameter C={log_reg.best_c}.\nIt achieved an AUC of {log_reg.best_auc} on the validation set, which was the best performance among all tested models.")
    with open("final_model_equation.txt", "w") as f:
        f.write(f"Final Model Equation: {equation}\n")
