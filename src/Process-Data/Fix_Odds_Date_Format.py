import sqlite3
import pandas as pd
import os

# Path to the database
ODDS_DB_PATH = r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\OddsData.sqlite"

# Connect to the database
if not os.path.exists(ODDS_DB_PATH):
    print(f"Database file not found at {ODDS_DB_PATH}")
    exit(1)

odds_con = sqlite3.connect(ODDS_DB_PATH)

# Fetch data from the table
try:
    odds_df = pd.read_sql_query('SELECT * FROM "odds_2024-25"', odds_con, index_col="index")
    print(f"Fetched {len(odds_df)} rows from odds_2024-25.")
except Exception as e:
    print(f"Error fetching data: {e}")
    odds_con.close()
    exit(1)

# Process the Date column
if not odds_df.empty:
    try:
        # Original logic for Date column processing
        arr = odds_df['Date'].apply(lambda x: pd.to_datetime(x, errors='coerce').strftime('%Y-%m-%d') if pd.notnull(x) else None)
        print(f"Length of arr: {len(arr)}")

        # Check length match
        if len(arr) != len(odds_df):
            raise ValueError(f"Length of arr ({len(arr)}) does not match length of odds_df ({len(odds_df)})")

        # Assign the formatted dates back to the 'Date' column
        odds_df['Date'] = arr
        print("Updated Date column successfully.")
    except Exception as e:
        print(f"Error updating Date column: {e}")
else:
    print("No data in odds_2024-25 table to process.")

# Save the updated DataFrame back to the table
try:
    odds_df.to_sql("odds_2024-25", odds_con, if_exists='replace', index=True)
    print(f"Updated table 'odds_2024-25' successfully.")
except Exception as e:
    print(f"Error saving updated table: {e}")

# Close the connection
odds_con.close()
