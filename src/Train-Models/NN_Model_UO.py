import sqlite3
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# Paths for logs and models
base_dir = os.path.abspath(r"C:\nba\NBA-Machine-Learning-Sports-Betting")
logs_dir = os.path.join(base_dir, "Logs")
models_dir = os.path.join(base_dir, "Models")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Unique directory for TensorBoard logs
current_time = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(logs_dir, current_time)
os.makedirs(log_dir, exist_ok=True)

# Callbacks
tensorboard = TensorBoard(log_dir=log_dir)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
mcp_save = ModelCheckpoint(
    os.path.join(models_dir, 'best_model.h5'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Load dataset
db_path = os.path.join(base_dir, "Data", "dataset.sqlite")
try:
    con = sqlite3.connect(db_path)
    print(f"Connected to SQLite database at {db_path}")
    dataset = pd.read_sql_query("SELECT * FROM dataset_combined", con)
    con.close()
    print(f"Loaded dataset with shape: {dataset.shape}")
except sqlite3.OperationalError as e:
    raise FileNotFoundError(f"Error accessing database at {db_path}: {e}")

# Preprocessing
print("Starting data preprocessing...")
try:
    # Target variable for classification
    target = dataset['OU-Cover']
    dataset.drop(
        ['OU-Cover', 'TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1', 'OU'],
        axis=1,
        inplace=True,
        errors='ignore'
    )
    # Convert to numeric, handle missing values
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset.fillna(0, inplace=True)
    print(f"Data preprocessing complete. Shape: {dataset.shape}")
except KeyError as e:
    raise KeyError(f"Column not found during preprocessing: {e}")

# Feature scaling
scaler = StandardScaler()
x_data = scaler.fit_transform(dataset.values)
y_data = np.where(target > 0, 1, 0)  # Binary classification labels
x_data, y_data = shuffle(x_data, y_data, random_state=42)

# Model definition
print("Building the model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(x_data.shape[1],)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),  # Regularization
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model compiled successfully.")

# Model training
print("Starting training...")
history = model.fit(
    x_data, y_data,
    epochs=50,
    validation_split=0.1,
    batch_size=32,
    callbacks=[tensorboard, earlyStopping, mcp_save]
)
print("Model training complete.")

# Save the final model in both `SavedModel` and `.h5` formats
saved_model_dir = os.path.join(models_dir, "SavedModel-Final")
os.makedirs(saved_model_dir, exist_ok=True)
model.save(saved_model_dir, save_format='tf')  # Save in TensorFlow's SavedModel format
print(f"Final model saved in SavedModel format to: {saved_model_dir}")

h5_model_path = os.path.join(models_dir, "final_model.h5")
model.save(h5_model_path)  # Save in HDF5 format
print(f"Final model saved in HDF5 format to: {h5_model_path}")

# Evaluate the model
loss, accuracy = model.evaluate(x_data, y_data)
print(f"Final Training Loss: {loss:.4f}, Training Accuracy: {accuracy:.4f}")
print("Done")
