import os
import sqlite3
import pandas as pd

# Paths to the databases
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATASET_DB_PATH = os.path.join(PROJECT_ROOT, "Data/dataset.sqlite")

# Table names
OLD_DATASET = "dataset_2012-24"
NEW_DATASET = "dataset_2012-25"
COMBINED_DATASET = "dataset_combined"

# Combine datasets
def combine_datasets(old_table, new_table, output_table, db_path):
    with sqlite3.connect(db_path) as con:
        try:
            print(f"Loading {old_table}...")
            old_data = pd.read_sql_query(f'SELECT * FROM "{old_table}"', con)

            print(f"Loading {new_table}...")
            new_data = pd.read_sql_query(f'SELECT * FROM "{new_table}"', con)

            # Drop the TEAM_ID column from the new dataset if it exists
            if "TEAM_ID" in new_data.columns:
                print("Dropping TEAM_ID column from new dataset...")
                new_data = new_data.drop(columns=["TEAM_ID"])

            # Combine datasets
            print("Combining datasets...")
            combined_data = pd.concat([old_data, new_data], ignore_index=True)

            # Save the combined dataset
            print(f"Saving combined dataset to {output_table}...")
            combined_data.to_sql(output_table, con, if_exists="replace", index=False)

            print(f"Combined dataset {output_table} created successfully.")

        except Exception as e:
            print(f"Error combining datasets: {e}")

# Execute the combination process
if __name__ == "__main__":
    combine_datasets(OLD_DATASET, NEW_DATASET, COMBINED_DATASET, DATASET_DB_PATH)
