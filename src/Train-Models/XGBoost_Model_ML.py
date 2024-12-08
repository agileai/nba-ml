import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

# Paths and configuration
base_dir = r"C:\nba\NBA-Machine-Learning-Sports-Betting"
db_path = os.path.join(base_dir, "Data", "dataset.sqlite")
models_dir = os.path.join(base_dir, "Models")
os.makedirs(models_dir, exist_ok=True)

# Check if the database file exists
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Error: Database file not found at {db_path}")

# Open connection to the database
con = sqlite3.connect(db_path)

# Define the dataset table name
dataset_table = "dataset_combined"

# Try reading the dataset
try:
    data = pd.read_sql_query(f"SELECT * FROM \"{dataset_table}\"", con, index_col="index")
    print(f"Loaded dataset from '{dataset_table}' with shape {data.shape}")
except Exception as e:
    con.close()
    raise RuntimeError(f"Error reading the dataset: {e}")

con.close()

# Ensure the dataset has the required columns
required_columns = ['Home-Team-Win']
if not all(col in data.columns for col in required_columns):
    raise KeyError(f"Error: Required columns {required_columns} are missing from the dataset.")

# Process the data
print("Starting data preprocessing...")
margin = data['Home-Team-Win']
data.drop(
    ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'],
    axis=1, inplace=True, errors='ignore'
)

# Identify non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=['float64', 'int64']).columns
if len(non_numeric_columns) > 0:
    print(f"Non-numeric columns detected: {non_numeric_columns}")

# Convert all columns to numeric, coercing errors
data = data.apply(pd.to_numeric, errors='coerce')

# Handle missing values
if data.isnull().sum().sum() > 0:
    print("Handling missing values...")
    data.fillna(0, inplace=True)  # Replace missing values with 0

print(f"Data preprocessing complete. Shape: {data.shape}")

# Convert data to NumPy array for XGBoost
data = data.values
data = data.astype(float)

# Initialize a list to store accuracy results
acc_results = []

# Train the model 300 times using different training-test splits
highest_accuracy = 0
best_model_path = ""

for x in tqdm(range(300)):
    x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=0.1, random_state=x)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

    param = {
        'max_depth': 3,
        'eta': 0.01,
        'objective': 'multi:softprob',
        'num_class': 2
    }
    epochs = 750

    # Train the XGBoost model
    model = xgb.train(param, train, epochs)

    # Make predictions
    predictions = model.predict(test)
    y = [np.argmax(z) for z in predictions]

    # Calculate accuracy
    acc = round(accuracy_score(y_test, y) * 100, 1)
    acc_results.append(acc)

    # Save the model if it achieves the highest accuracy so far
    if acc > highest_accuracy:
        highest_accuracy = acc
        best_model_path = os.path.join(models_dir, f"XGBoost_{acc}%_ML-4.json")
        model.save_model(best_model_path)
        print(f"New best model saved with accuracy: {acc}% at {best_model_path}")

print("Training completed.")
print(f"Highest Accuracy Achieved: {highest_accuracy}%")
print(f"Best Model Saved at: {best_model_path}")
