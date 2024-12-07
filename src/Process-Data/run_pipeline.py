import os
import sqlite3
import toml
import pandas as pd
from datetime import datetime, date, timedelta
from sbrscrape import Scoreboard
import random
import time

# Define paths
CONFIG_PATH = r"C:\nba\NBA-Machine-Learning-Sports-Betting\config.toml"
TEAM_DATA_PATH = r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\TeamData.sqlite"
ODDS_DATA_PATH = r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\OddsData.sqlite"

# Load configuration
config = toml.load(CONFIG_PATH)

# Helper function to fetch and append data
def fetch_team_data():
    print("Starting Team Data Fetch...")
    con = sqlite3.connect(TEAM_DATA_PATH)
    try:
        for key, value in config['get-data'].items():
            date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
            end_date = min(datetime.strptime(value['end_date'], "%Y-%m-%d").date(), date.today())
            
            while date_pointer <= end_date:
                print(f"Fetching team data for {date_pointer}...")
                # Add your API fetch logic here
                # Mocked data for simplicity
                data = pd.DataFrame({
                    'Date': [date_pointer],
                    'Team': ['Mock Team'],
                    'Points': [random.randint(80, 120)]
                })
                data.to_sql(key, con, if_exists="append", index=False)
                date_pointer += timedelta(days=1)
                time.sleep(random.uniform(1, 2))
    except Exception as e:
        print(f"Error fetching team data: {e}")
    finally:
        con.close()
    print("Team Data Fetch Complete.")

def fetch_odds_data():
    print("Starting Odds Data Fetch...")
    con = sqlite3.connect(ODDS_DATA_PATH)
    try:
        for key, value in config['get-odds-data'].items():
            date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
            end_date = min(datetime.strptime(value['end_date'], "%Y-%m-%d").date(), date.today())
            
            while date_pointer <= end_date:
                print(f"Fetching odds data for {date_pointer}...")
                try:
                    sb = Scoreboard(date=date_pointer)
                    if not hasattr(sb, "games") or not sb.games:
                        print(f"No games found for {date_pointer}, skipping.")
                        date_pointer += timedelta(days=1)
                        continue

                    # Mocked data
                    data = pd.DataFrame({
                        'Date': [date_pointer],
                        'Home': ['Mock Home Team'],
                        'Away': ['Mock Away Team'],
                        'Odds': [random.uniform(1.5, 3.5)]
                    })
                    data.to_sql(key, con, if_exists="append", index=False)
                except Exception as e:
                    print(f"Error fetching odds data for {date_pointer}: {e}")
                date_pointer += timedelta(days=1)
                time.sleep(random.uniform(1, 2))
    except Exception as e:
        print(f"Error fetching odds data: {e}")
    finally:
        con.close()
    print("Odds Data Fetch Complete.")

def update_combined_dataset():
    print("Starting Combined Dataset Update...")
    con_team = sqlite3.connect(TEAM_DATA_PATH)
    con_odds = sqlite3.connect(ODDS_DATA_PATH)
    try:
        combined_data = []
        for season in range(2012, 2025):  # 2012-24 seasons
            team_data = pd.read_sql_query(f"SELECT * FROM get_data.{season}-{season + 1}", con_team)
            odds_data = pd.read_sql_query(f"SELECT * FROM get_odds_data.{season}-{season + 1}", con_odds)
            combined = pd.merge(team_data, odds_data, on="Date", how="inner")
            combined_data.append(combined)
        
        # Combine all seasons into a single DataFrame
        final_dataset = pd.concat(combined_data, ignore_index=True)
        # Save the combined dataset
        final_dataset.to_csv(r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\FinalDataset.csv", index=False)
    except Exception as e:
        print(f"Error updating combined dataset: {e}")
    finally:
        con_team.close()
        con_odds.close()
    print("Combined Dataset Update Complete.")

# Run the pipeline
if __name__ == "__main__":
    print("Starting Data Pipeline...")
    fetch_team_data()
    fetch_odds_data()
    update_combined_dataset()
    print("Data Pipeline Complete.")
