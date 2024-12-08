import sqlite3
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Absolute path to the SQLite database
db_path = r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\dataset.sqlite"

# Check if the database file exists
if not os.path.exists(db_path):
    print(f"Error: Database file not found at {db_path}")
    exit(1)

# Open connection to the database
con = sqlite3.connect(db_path)
dataset = "dataset_2012-24_new"
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Validate data
if data.isnull().sum().sum() > 0:
    print("Error: Dataset contains missing values.")
    exit(1)

OU = data['OU-Cover']
total = data['OU']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)
data['OU'] = np.asarray(total)
data = data.astype(np.float32).values  # Optimize memory usage

# Validate target variable
print(f"Unique classes in OU-Cover: {np.unique(OU)}")
OU = np.asarray(OU, dtype=np.int32)  # Ensure target values are integers

# Ensure output directory exists
output_dir = '../../Models/'
os.makedirs(output_dir, exist_ok=True)

# XGBoost parameters for CPU usage
param = {
    'max_depth': 4,               # Reduced for faster training
    'eta': 0.1,                  # Simplified learning rate
    'objective': 'multi:softprob',  # Multi-class classification
    'num_class': len(np.unique(OU)),  # Dynamically set number of classes
    'tree_method': 'hist',       # Use histogram-based method for CPU
    'subsample': 0.8,            # Subsample ratio of the training instances
    'colsample_bytree': 0.8,     # Subsample ratio of columns for each tree
    'device': 'cpu'              # Explicitly set to use CPU
}
early_stopping_rounds = 10  # Early stopping criteria
num_boost_round = 100  # Reduced boosting rounds for debugging
acc_results = []

# Training loop
for x in tqdm(range(10)):  # Reduced iterations for debugging
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=0.1, random_state=x)

    # Validate data split
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Create DMatrix for XGBoost
    try:
        train = xgb.DMatrix(x_train, label=y_train)
        test = xgb.DMatrix(x_test)
    except ValueError as e:
        print(f"Error creating DMatrix: {e}")
        continue

    # Validate labels
    print(f"Unique training labels: {np.unique(y_train)}")
    print(f"Unique test labels: {np.unique(y_test)}")

    # Train the model with early stopping
    print("Starting training...")
    try:
        model = xgb.train(
            param,
            train,
            num_boost_round=num_boost_round,
            evals=[(train, 'train'), (test, 'test')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )
        print("Model training complete.")

        # Make predictions
        predictions = model.predict(test)
        if predictions.size == 0:
            print("Predictions are empty. Skipping this iteration.")
            continue

        y = [np.argmax(z) for z in predictions]

        # Calculate accuracy
        acc = round(accuracy_score(y_test, y) * 100, 1)
        print(f"Iteration {x}: Accuracy = {acc}%")
        acc_results.append(acc)

        # Save the best model
        if acc == max(acc_results):
            model.save_model(f'{output_dir}/XGBoost_{acc}%_UO-9.json')

    except xgb.core.XGBoostError as e:
        print(f"Error during training: {e}")
        continue

if acc_results:
    print("Training complete. Best accuracy: {:.1f}%".format(max(acc_results)))
else:
    print("No successful training iterations.")
