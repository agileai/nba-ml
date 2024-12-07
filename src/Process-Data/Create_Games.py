import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import toml
from datetime import datetime

# Add the project root directory to sys.path for module resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Import dictionaries for team indexing
from src.Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14, team_index_current

# Load configuration
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.toml")
config = toml.load(CONFIG_PATH)

# Initialize variables
scores, win_margin, OU, OU_Cover, games = [], [], [], [], []
days_rest_away, days_rest_home = [], []

# Database paths
TEAM_DB_PATH = os.path.join(PROJECT_ROOT, "Data/TeamData.sqlite")
ODDS_DB_PATH = os.path.join(PROJECT_ROOT, "Data/OddsData.sqlite")
OUTPUT_DB_PATH = os.path.join(PROJECT_ROOT, "Data/dataset.sqlite")

# Get the current date
current_date = datetime.today().strftime("%Y-%m-%d")

# Process data
with sqlite3.connect(TEAM_DB_PATH) as teams_con, sqlite3.connect(ODDS_DB_PATH) as odds_con:
    for key, value in config['create-games'].items():
        print(f"Processing season: {key}")
        odds_df = pd.read_sql_query(f"SELECT * FROM \"odds_{key}_new\"", odds_con, index_col="index")

        for row in odds_df.itertuples():
            home_team, away_team, date = row[2], row[3], row[1]

            if date > current_date:
                print(f"Skipping future date: {date}")
                continue

            try:
                # Check for table existence before querying
                cursor = teams_con.cursor()
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{date}'")
                if cursor.fetchone() is None:
                    print(f"Table for date {date} does not exist. Skipping.")
                    continue

                team_df = pd.read_sql_query(f"SELECT * FROM \"{date}\"", teams_con, index_col="index")
            except Exception as e:
                print(f"Error fetching data for date {date}: {e}")
                continue

            if len(team_df.index) == 30:  # Ensure valid team data
                scores.append(row[8])
                OU.append(row[4])
                days_rest_home.append(row[10])
                days_rest_away.append(row[11])

                # Determine win margin
                win_margin.append(1 if row[9] > 0 else 0)

                # Determine OU cover
                if row[8] < row[4]:
                    OU_Cover.append(0)
                elif row[8] > row[4]:
                    OU_Cover.append(1)
                else:
                    OU_Cover.append(2)

                # Map teams based on season
                if key == '2007-08':
                    home_team_series = team_df.iloc[team_index_07.get(home_team)]
                    away_team_series = team_df.iloc[team_index_07.get(away_team)]
                elif key in ['2008-09', '2009-10', '2010-11', '2011-12']:
                    home_team_series = team_df.iloc[team_index_08.get(home_team)]
                    away_team_series = team_df.iloc[team_index_08.get(away_team)]
                elif key == '2012-13':
                    home_team_series = team_df.iloc[team_index_12.get(home_team)]
                    away_team_series = team_df.iloc[team_index_12.get(away_team)]
                elif key == '2013-14':
                    home_team_series = team_df.iloc[team_index_13.get(home_team)]
                    away_team_series = team_df.iloc[team_index_13.get(away_team)]
                elif key in ['2022-23', '2023-24', '2024-25']:
                    home_team_series = team_df.iloc[team_index_current.get(home_team)]
                    away_team_series = team_df.iloc[team_index_current.get(away_team)]
                else:
                    home_team_series = team_df.iloc[team_index_14.get(home_team)]
                    away_team_series = team_df.iloc[team_index_14.get(away_team)]

                # Concatenate team data
                game = pd.concat([home_team_series, away_team_series.rename(
                    index={col: f"{col}.1" for col in team_df.columns.values}
                )])
                games.append(game)

# Compile season data
season = pd.concat(games, ignore_index=True, axis=1).T
season['Score'] = np.asarray(scores)
season['Home-Team-Win'] = np.asarray(win_margin)
season['OU'] = np.asarray(OU)
season['OU-Cover'] = np.asarray(OU_Cover)
season['Days-Rest-Home'] = np.asarray(days_rest_home)
season['Days-Rest-Away'] = np.asarray(days_rest_away)

# Fix types
for field in season.columns.values:
    if 'TEAM_' in field or 'Date' in field or field not in season:
        continue
    season[field] = season[field].astype(float)

# Save to dataset.sqlite
with sqlite3.connect(OUTPUT_DB_PATH) as con:
    season.to_sql("dataset_2012-25", con, if_exists="replace")
    print("Updated table: dataset_2012-25")
