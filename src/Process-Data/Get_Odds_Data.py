import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta, date

import pandas as pd
import toml
from sbrscrape import Scoreboard

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# Hardcoded path to config.toml
CONFIG_PATH = r"C:\nba\NBA-Machine-Learning-Sports-Betting\config.toml"
config = toml.load(CONFIG_PATH)

# Hardcoded path to OddsData.sqlite
ODDS_DATA_PATH = r"C:\nba\NBA-Machine-Learning-Sports-Betting\Data\OddsData.sqlite"
con = sqlite3.connect(ODDS_DATA_PATH)

sportsbook = 'fanduel'
df_data = []

# Only fetch 2024-25 odds
for key, value in config['get-odds-data'].items():
    if key != "2024-25":
        continue  # Skip other seasons

    date_pointer = datetime.strptime(value['start_date'], "%Y-%m-%d").date()
    end_date = min(datetime.strptime(value['end_date'], "%Y-%m-%d").date(), date.today())
    teams_last_played = {}

    while date_pointer <= end_date:
        print("Getting odds data: ", date_pointer)
        try:
            sb = Scoreboard(date=date_pointer)
            if not hasattr(sb, "games") or not sb.games:
                print(f"No games found for {date_pointer}, skipping.")
                date_pointer += timedelta(days=1)
                continue

            for game in sb.games:
                # Skip if odds data is incomplete
                try:
                    df_data.append({
                        'Date': date_pointer,
                        'Home': game['home_team'],
                        'Away': game['away_team'],
                        'OU': game['total'][sportsbook],
                        'Spread': game['away_spread'][sportsbook],
                        'ML_Home': game['home_ml'][sportsbook],
                        'ML_Away': game['away_ml'][sportsbook],
                        'Points': game['away_score'] + game['home_score'],
                        'Win_Margin': game['home_score'] - game['away_score'],
                        'Days_Rest_Home': (date_pointer - teams_last_played.get(game['home_team'], date_pointer)).days,
                        'Days_Rest_Away': (date_pointer - teams_last_played.get(game['away_team'], date_pointer)).days,
                    })
                    teams_last_played[game['home_team']] = date_pointer
                    teams_last_played[game['away_team']] = date_pointer
                except KeyError:
                    print(f"Missing data for {game}, skipping.")
        except Exception as e:
            print(f"Error fetching data for {date_pointer}: {e}")

        # Ensure date_pointer advances
        date_pointer += timedelta(days=1)
        time.sleep(random.uniform(1, 3))  # Random delay to avoid API throttling

# Save to SQLite
df = pd.DataFrame(df_data)
df.to_sql(key, con, if_exists="replace")
con.close()
