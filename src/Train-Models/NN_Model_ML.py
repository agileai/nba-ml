import sqlite3
import time
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import joblib

# Paths
current_time = str(time.time())
logs_dir = os.path.abspath(f'../../Logs/{current_time}')
model_dir = os.path.abspath(f'../../Models/Trained-Model-ML-{current_time}')
scaler_path = os.path.abspath('../../Models/scaler.save')

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_dir), exist_ok=True)

# Callbacks
tensorboard = TensorBoard(log_dir=logs_dir)
earlyStopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')  # Reduced patience to prevent overfitting
mcp_save = ModelCheckpoint(model_dir, save_best_only=True, monitor='val_loss', mode='min')

# Load data
dataset = "dataset_combined"
db_path = r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\dataset.sqlite"
con = sqlite3.connect(db_path)
print(f"Connected to SQLite database at {db_path}")

data = pd.read_sql_query(f"SELECT * FROM \"{dataset}\"", con, index_col="index")
con.close()
print(f"Raw data loaded from SQLite with shape: {data.shape}")
print(data.head())

# Preprocess data
try:
    scores = data['Score']
    margin = data['Home-Team-Win']
    data.drop(
        ['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU', 'OU-Cover'],
        axis=1,
        inplace=True,
        errors='ignore'
    )
    print(f"Data shape after dropping unnecessary columns: {data.shape}")
except KeyError as e:
    print(f"Column dropping error: {e}")

# Sanitize and handle missing values
data = data.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, replacing invalid values with NaN
print(f"Data shape after numeric conversion: {data.shape}")

if data.isnull().values.any():
    print("Missing values detected. Replacing NaNs with 0.")
    data.fillna(0, inplace=True)  # Replace NaN with 0

# Check if data is empty
if data.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Please check your data source and preprocessing steps.")

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(data.values)
y_train = np.where(margin > 0, 1, 0)  # Convert margin to binary labels
print("Data scaled successfully.")

# Save scaler for future use
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# Shuffle data
x_train, y_train = shuffle(x_train, y_train, random_state=42)

# Build model with dropout for regularization
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu6),
    tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(256, activation=tf.nn.relu6),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu6),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# Train model with validation split
model.fit(
    x_train, y_train,
    epochs=50,
    validation_split=0.2,  # Increased validation split to 20%
    batch_size=32,
    callbacks=[tensorboard, earlyStopping, mcp_save]
)
print("Model training complete.")

# Save model in .keras format
model_path = os.path.join(model_dir, "final_model.keras")
model.save(model_path, save_format='keras')
print(f"Model saved to: {model_path}")

# Evaluate model
loss, accuracy = model.evaluate(x_train, y_train)
print(f"Final Training Loss: {loss}, Training Accuracy: {accuracy}")

print('Done')
