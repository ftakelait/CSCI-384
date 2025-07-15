import pandas as pd
import numpy as np

def load_and_discretize_weather_data(csv_path):
    """
    Load weather data and discretize into observation symbols for HMM.
    
    Weather discretization rules:
    - Rainy: if PRCP > 0.01 inches
    - Snowy: if SNOW > 0.1 inches
    - Sunny: if TMAX > 75°F and PRCP == 0
    - Cloudy: if 40°F < TMAX <= 75°F and PRCP == 0
    - Cold: if TMAX <= 40°F
    
    Args:
        csv_path (str): Path to the weather CSV file.
    Returns:
        pd.DataFrame: Original weather dataframe
        list: List of observation symbols (as string labels)
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Discretize observations
    obs_labels = []
    for index, row in df.iterrows():
        # Rainy classification: if PRCP > 0.01
        if row['PRCP'] > 0.01:
            obs_labels.append('Rainy')
        # Snowy classification: if SNOW > 0.1
        elif row['SNOW'] > 0.1:
            obs_labels.append('Snowy')
        # Sunny classification: if TMAX > 75 and PRCP == 0
        elif row['TMAX'] > 75 and row['PRCP'] == 0:
            obs_labels.append('Sunny')
        # Cloudy classification: if 40 < TMAX <= 75 and PRCP == 0
        elif 40 < row['TMAX'] <= 75 and row['PRCP'] == 0:
            obs_labels.append('Cloudy')
        # Cold classification: if TMAX <= 40
        elif row['TMAX'] <= 40:
            obs_labels.append('Cold')
        else:
            # Fallback: assign 'Cloudy' for any edge cases
            obs_labels.append('Cloudy')
    
    return df, obs_labels 