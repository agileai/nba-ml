import copy
import numpy as np
import tensorflow as tf
from colorama import Fore, Style, init, deinit
from keras.models import load_model
import sys
import os

# Add the src directory to the Python path for module imports
src_path = r'C:\nba\NBA-Machine-Learning-Sports-Betting\src'
if src_path not in sys.path:
    sys.path.append(src_path)

# Check if the path was added successfully
print("Current sys.path:", sys.path)

# Error handling for module imports
try:
    from src.Utils import Expected_Value
    from src.Utils import Kelly_Criterion as kc
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Initialize colorama for colored text output
init()

# Load pre-trained models
model = load_model(r'C:\nba\NBA-Machine-Learning-Sports-Betting\Models\Trained-Model-ML-1733602644.936518')
ou_model = load_model(r'C:\nba\NBA-Machine-Learning-Sports-Betting\Models\SavedModel-Final')

# Function to predict game outcomes (ML)
def predict_ml(data):
    ml_predictions_array = []
    for row in data:
        ml_predictions_array.append(model.predict(np.array([row])))
    return ml_predictions_array

# Function to predict over/under outcomes (OU)
def predict_ou(data):
    ou_predictions_array = []
    for row in data:
        ou_predictions_array.append(ou_model.predict(np.array([row])))
    return ou_predictions_array

# Main runner function to predict game outcomes
def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    print("Running predictions...")

    # Predict ML outcomes (moneyline)
    ml_predictions_array = predict_ml(data)

    # Copy the frame and add OU values
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo)

    # Prepare the data for OU prediction
    data = frame_uo.values
    data = data.astype(float)
    data = tf.keras.utils.normalize(data, axis=1)

    # Predict OU outcomes (over/under)
    ou_predictions_array = predict_ou(data)

    count = 0
    # Loop through all games and print predictions
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]

        # Print the prediction for the home and away teams
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print_prediction(home_team, away_team, winner_confidence, un_confidence, "UNDER", todays_games_uo[count])
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print_prediction(home_team, away_team, winner_confidence, un_confidence, "OVER", todays_games_uo[count])
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0][0] * 100, 1)
                print_prediction(home_team, away_team, winner_confidence, un_confidence, "UNDER", todays_games_uo[count])
            else:
                un_confidence = round(ou_predictions_array[count][0][1] * 100, 1)
                print_prediction(home_team, away_team, winner_confidence, un_confidence, "OVER", todays_games_uo[count])

        count += 1

    # Calculate Expected Value & Kelly Criterion
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")

    calculate_expected_value_and_kelly(games, ml_predictions_array, home_team_odds, away_team_odds, kelly_criterion)

    # Deinitialize colorama
    deinit()

# Helper function to print the formatted predictions
def print_prediction(home_team, away_team, winner_confidence, un_confidence, bet_type, todays_game_uo):
    """ Helper function to print formatted predictions """
    print(Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + 
          ' vs ' + Fore.RED + away_team + Style.RESET_ALL + f': {Fore.MAGENTA}{bet_type} ' +
          Style.RESET_ALL + str(todays_game_uo) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)

# Function to calculate Expected Value and Kelly Criterion
def calculate_expected_value_and_kelly(games, ml_predictions_array, home_team_odds, away_team_odds, kelly_criterion):
    """ Helper function to calculate expected value and Kelly Criterion """
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        if home_team_odds[count] and away_team_odds[count]:
            ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], int(home_team_odds[count])))
            ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], int(away_team_odds[count])))

        expected_value_colors = {
            'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
            'away_color': Fore.GREEN if ev_away > 0 else Fore.RED
        }

        bankroll_descriptor = ' Fraction of Bankroll: '
        bankroll_fraction_home = bankroll_descriptor + str(kc.calculate_kelly_criterion(home_team_odds[count], ml_predictions_array[count][0][1])) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(kc.calculate_kelly_criterion(away_team_odds[count], ml_predictions_array[count][0][0])) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))
        count += 1
