from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
#import pandas as pd
#import numpy as np

# Load Iris dataset
iris = load_iris()
X = iris.data
y_class = iris.target
y_regress = X[:, 3]

# Train-test split
X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.3, random_state=42)
_, _, y_regress_train, y_regress_test = train_test_split(X, y_regress, test_size=0.3, random_state=42)


# Classification Model
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_class_train)
y_class_pred = clf.predict(X_test)

print("=== Classification (Logistic Regression) ===")
print("Accuracy:", accuracy_score(y_class_test, y_class_pred))
print("Classification Report:\n", classification_report(y_class_test, y_class_pred))


# Regression Model
reg = LinearRegression()
reg.fit(X_train, y_regress_train)
y_regress_pred = reg.predict(X_test)

print("\n=== Regression (Predicting Petal Width) ===")
print("MSE:", mean_squared_error(y_regress_test, y_regress_pred))

# Neural Network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=3000, random_state=42, solver='lbfgs')
mlp.fit(X_train, y_class_train)
y_mlp_pred = mlp.predict(X_test)

print("\n=== Neural Network (MLPClassifier) ===")
print("Accuracy:", accuracy_score(y_class_test, y_mlp_pred))
print("Classification Report:\n", classification_report(y_class_test, y_mlp_pred))
