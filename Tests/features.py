import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'c:\Users\jdchi\Downloads\dataset_combined.csv')

# Step 1: Select numeric columns and calculate correlation
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Check for correlations with Home-Team-Win
correlations = numeric_data.corr()['Home-Team-Win'].sort_values(ascending=False)
print("Correlation with Home-Team-Win:\n", correlations)

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Step 2: Feature importance using Random Forest
target = 'Home-Team-Win'
features = ['OU-Cover', 'W_PCT', 'Days-Rest-Home', 'Days-Rest-Away', 'FG_PCT']

# Define X and y
X = data[features]
y = data[target]

# Handle missing values
X = X.fillna(X.mean())

# Train Random Forest for feature importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# Extract feature importances
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("Feature Importances:\n", importances)

# Visualize feature importance
importances.plot(kind='bar', title="Feature Importance", color='skyblue')
plt.show()

# Step 3: Recursive Feature Elimination (RFE) with Logistic Regression
rfe_model = LogisticRegression(max_iter=1000)
selector = RFE(estimator=rfe_model, n_features_to_select=3, step=1)
selector = selector.fit(X, y)

# Selected features
selected_features = X.columns[selector.support_]
print("Selected Features with RFE:\n", selected_features)

# Train a logistic regression model with selected features
X_selected = X[selected_features]
rfe_model.fit(X_selected, y)

# Visualize results
plt.figure(figsize=(8, 6))
sns.countplot(x=data['Home-Team-Win'])
plt.title("Home-Team-Win Distribution")
plt.show()

# Summary
print("Script completed! Refined features and importance analysis are above.")
