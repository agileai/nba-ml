import re
import sqlite3
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os

# Define a function to parse and correct the date format
def get_date(date_string):
    try:
        return datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_string}")

# Database connection setup
ODDS_DB_PATH = r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\OddsData.sqlite"
if not os.path.exists(ODDS_DB_PATH):
    print(f"Error: Database file not found at {ODDS_DB_PATH}")
    exit(1)

# Connect to the SQLite database
try:
    con = sqlite3.connect(ODDS_DB_PATH)
    print("Database connected successfully.")
except sqlite3.OperationalError as e:
    print(f"Error connecting to database: {e}")
    exit(1)

# Define the datasets to process
datasets = ["odds_2024-25"]  # Only process the 2024-25 season

# Process each dataset
for dataset in tqdm(datasets):
    try:
        # Read data from the raw dataset
        raw_data = pd.read_sql_query(f"SELECT * FROM \"{dataset}\"", con, index_col="index")
        print(f"Processing dataset: {dataset}")

        # Create a copy of the raw data for the new dataset
        fixed_dataset = f"{dataset}_new"
        data = raw_data.copy()

        # Initialize tracking of teams' last played dates
        teams_last_played = {}

        # Iterate through rows to calculate days rested
        for index, row in data.iterrows():
            # Ensure 'Home' and 'Away' columns exist
            if 'Home' not in row or 'Away' not in row or 'Date' not in row:
                print(f"Skipping row {index} due to missing data.")
                continue

            # Calculate days rested for the home team
            home_team = row['Home']
            if home_team not in teams_last_played:
                teams_last_played[home_team] = get_date(row['Date'])
                home_games_rested = 10
            else:
                current_date = get_date(row['Date'])
                home_games_rested = (current_date - teams_last_played[home_team]).days
                home_games_rested = max(0, min(home_games_rested, 9))  # Cap at 9 days
                teams_last_played[home_team] = current_date

            # Calculate days rested for the away team
            away_team = row['Away']
            if away_team not in teams_last_played:
                teams_last_played[away_team] = get_date(row['Date'])
                away_games_rested = 10
            else:
                current_date = get_date(row['Date'])
                away_games_rested = (current_date - teams_last_played[away_team]).days
                away_games_rested = max(0, min(away_games_rested, 9))  # Cap at 9 days
                teams_last_played[away_team] = current_date

            # Update the DataFrame with calculated values
            data.at[index, 'Days_Rest_Home'] = home_games_rested
            data.at[index, 'Days_Rest_Away'] = away_games_rested

        # Save the updated dataset back to a new table in the database
        data.to_sql(fixed_dataset, con, if_exists='replace', index=True)
        print(f"Dataset '{fixed_dataset}' created successfully.")

    except Exception as e:
        print(f"Error processing dataset '{dataset}': {e}")

# Close the database connection
con.close()
print("Database connection closed.")