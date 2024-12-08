import os
import sqlite3
import numpy as np
import pandas as pd

# Database path
DB_PATH = os.path.abspath("C:/nba/NBA-Machine-Learning-Sports-Betting/Data/dataset.sqlite")

# Connect to the SQLite database
try:
    con = sqlite3.connect(DB_PATH)
    print(f"Connected to database at {DB_PATH}")
except sqlite3.OperationalError as e:
    print(f"Error connecting to database: {e}")
    sys.exit(1)

# Load the dataset
try:
    query = "SELECT * FROM cleaned_dataset"
    data = pd.read_sql_query(query, con)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    con.close()
    sys.exit(1)

# Inspect data
print("Initial data preview:")
print(data.head())
print("Data types:")
print(data.dtypes)

# Add a Score column (PTS + PTS.1)
if 'PTS' in data.columns and 'PTS.1' in data.columns:
    try:
        data['Score'] = data['PTS'] + data['PTS.1']
        print("Score column added successfully.")
    except Exception as e:
        print(f"Error adding Score column: {e}")
        con.close()
        sys.exit(1)
else:
    print("PTS or PTS.1 columns missing. Cannot calculate Score.")
    con.close()
    sys.exit(1)

# Drop unnecessary columns if they exist
columns_to_drop = ['TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU', 'OU-Cover']
data_columns = data.columns.tolist()
existing_columns_to_drop = [col for col in columns_to_drop if col in data_columns]

if existing_columns_to_drop:
    try:
        data.drop(existing_columns_to_drop, axis=1, inplace=True)
        print(f"Dropped columns: {existing_columns_to_drop}")
    except Exception as e:
        print(f"Error dropping columns: {e}")
        con.close()
        sys.exit(1)
else:
    print("No matching columns to drop.")

# Ensure Home-Team-Win column exists
if 'Home-Team-Win' not in data.columns:
    print("Error: 'Home-Team-Win' column is missing from the dataset.")
    con.close()
    sys.exit(1)

# Clean and process data
try:
    # Log missing expected columns
    expected_columns = ['Home-Team-Win', 'Score']
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        print(f"Warning: Missing columns in dataset: {missing_columns}")

    # Proceed with available numeric data
    numeric_data = data.select_dtypes(include=[np.number])

    # Handle conversion errors for non-numeric values
    numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    numeric_data = numeric_data.dropna()

    # Ensure all values are float
    numeric_data = numeric_data.astype(float)

    print("Data cleaned and converted to float successfully.")
except Exception as e:
    print(f"Error during data cleaning: {e}")
    con.close()
    sys.exit(1)

# Preview cleaned data
print("Cleaned data preview:")
print(numeric_data.head())

# Example: Save cleaned data back to the database for verification
try:
    numeric_data.to_sql("cleaned_dataset", con, if_exists="replace", index=False)
    print("Cleaned dataset saved to database as 'cleaned_dataset'.")
except Exception as e:
    print(f"Error saving cleaned dataset: {e}")

# Close the database connection
con.close()
print("Database connection closed.")
