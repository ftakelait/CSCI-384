"""
Main script for the Hidden Markov Model (HMM) Weather Assignment
Complete each section as instructed. Use dataset_utils and hmm_model for support.

This script implements a complete HMM workflow for weather prediction:
1. Data loading and preprocessing
2. HMM parameter estimation from real weather data
3. Filtering using the forward algorithm
4. Prediction of future weather states
5. Smoothing using the forward-backward algorithm
6. Evaluation and accuracy calculation

POINTS IN THIS FILE: 50

TODO: Complete all sections to achieve full points and >0.5 weather type accuracy
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from collections import Counter
from dataset_utils import load_and_discretize_weather_data
from hmm_model import HiddenMarkovModel

# === STEP 1: DATA LOADING AND PREPROCESSING (5 points) ===
# TODO: Load the weather data and discretize observations (3 points)
# THINK: What function should you use to load and process the weather data?
weather_df, obs_labels = load_and_discretize_weather_data('data/grand_forks_daily_weather_2019_2024.csv')
"""weather_df: pandas DataFrame containing the original weather data
obs_labels: list of string labels for each day's weather observation"""

# TODO: Define observation symbol mapping (2 points)
# THINK: What are the possible weather states? How do you map strings to numbers?
observation_symbols = ['Rainy', 'Snowy', 'Sunny', 'Cloudy', 'Cold']
"""observation_symbols: list of possible observed weather states"""
symbol_to_idx = {label: i for i, label in enumerate(observation_symbols)}
observations = [symbol_to_idx.get(label, 2) for label in obs_labels]  # Default to Sunny (2) for unknown labels
"""observations: list of integer indices for each observation label"""

# Print data loading summary
print(f"Dataset loaded: {len(weather_df)} days of weather data")
print(f"Observations created: {len(observations)} weather states")
print(f"Observation symbols: {observation_symbols}")
print(f"Unique labels in data: {set(obs_labels)}")

# === STEP 2: HMM MODEL SETUP (15 points) ===
# TODO: Define hidden states and parameter estimation (2 points)
# THINK: What underlying weather patterns could explain the observed weather?
hidden_states = ['Dry', 'Wet']
"""hidden_states: list of possible hidden weather states"""
state_to_idx = {'Dry': 0, 'Wet': 1}
"""state_to_idx: mapping from hidden state name to index"""
print(f"Hidden states: {hidden_states}")
print(f"State mapping: {state_to_idx}")

# TODO: Map observations to hidden states for parameter estimation (2 points)
# THINK: Which weather types belong to which hidden states?
hidden_state_seq = []
for label in obs_labels:
    if label in ['Rainy', 'Snowy']:
        hidden_state_seq.append(1)  # Wet state
    else:
        hidden_state_seq.append(0)  # Dry state
"""hidden_state_seq: list of integer indices for each hidden state (Wet=1, Dry=0)"""

# TODO: Estimate initial probabilities from the hidden state sequence (3 points)
# THINK: How do you estimate the probability of starting in each state?
initial_probs = np.zeros(len(hidden_states))
dry_count = sum(1 for state in hidden_state_seq if state == 0)
wet_count = sum(1 for state in hidden_state_seq if state == 1)
total_states = len(hidden_state_seq)
initial_probs[0] = dry_count / total_states  # Dry state probability
initial_probs[1] = wet_count / total_states  # Wet state probability
"""initial_probs: numpy array of initial state probabilities"""
print(f"Initial probabilities: {initial_probs}")

# TODO: Estimate transition probabilities from hidden state sequence (3 points)
# THINK: How do you count transitions between states and convert to probabilities?
transition_counts = np.zeros((2, 2))
for i in range(1, len(hidden_state_seq)):
    prev = hidden_state_seq[i-1]
    curr = hidden_state_seq[i]
    transition_counts[prev, curr] += 1

# Add small smoothing to avoid zero probabilities
transition_counts += 0.1
transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
"""transition_probs: numpy array of transition probabilities between hidden states"""
print(f"Transition probability matrix:\n{transition_probs}")

# TODO: Estimate emission probabilities from hidden state and observation sequence (5 points)
# THINK: How do you count which observations occur in which hidden states?
emission_counts = np.zeros((2, len(observation_symbols)))
for s, o in zip(hidden_state_seq, observations):
    emission_counts[s, o] += 1

# Add smoothing to avoid zero probabilities
# THINK: Why is smoothing important? What happens with zero probabilities?
smoothing_factor = 0.1
emission_counts += smoothing_factor

# Normalize to get probabilities
emission_probs = emission_counts / emission_counts.sum(axis=1, keepdims=True)
"""emission_probs: numpy array of emission probabilities for each hidden state and observation symbol"""
print(f"Emission probability matrix:\n{emission_probs}")

# TODO: Create HMM instance with estimated parameters (0 points - already done)
hmm = HiddenMarkovModel(hidden_states, observation_symbols, initial_probs, transition_probs, emission_probs)
"""hmm: HiddenMarkovModel instance for weather prediction"""
print("HMM instance created successfully")

# === STEP 3: FILTERING (FORWARD ALGORITHM) (5 points) ===
# TODO: Run the forward algorithm (5 points)
# THINK: Which method should you call to get filtered probabilities?
filtered_probs = hmm.forward(observations)
"""filtered_probs: numpy array of filtered state probabilities for each day"""
print(f"Filtering completed: {filtered_probs.shape if filtered_probs is not None else 'Not implemented'}")

# === STEP 4: PREDICTION (5 points) ===
# TODO: Predict the next state (5 points)
# THINK: How do you use the last filtered probabilities to predict the next state?
predicted_state = hmm.predict_next_state(filtered_probs[-1])
"""predicted_state: numpy array of predicted probabilities for the next state"""
print(f"Prediction completed: {predicted_state is not None}")
if predicted_state is not None:
    print(f"Predicted next state: {predicted_state}")

# === STEP 5: SMOOTHING (FORWARD-BACKWARD ALGORITHM) (5 points) ===
# TODO: Run the forward-backward algorithm (5 points)
# THINK: Which method should you call to get smoothed probabilities?
smoothed_probs = hmm.forward_backward(observations)
"""smoothed_probs: numpy array of smoothed state probabilities for each day"""
print(f"Smoothing completed: {smoothed_probs.shape if smoothed_probs is not None else 'Not implemented'}")

# === STEP 6: EVALUATION AND INTERPRETATION (15 points) ===
# TODO: Calculate accuracy and evaluation metrics (10 points)
# THINK: How do you compare predictions with actual observations?

# Method 1: Basic category accuracy (Dry/Wet) - 3 points
accuracy = 0.0
correct_predictions = 0
total_predictions = min(len(observations), len(filtered_probs))

category_correct = 0
for i in range(total_predictions):
    # Get the most likely hidden state (Dry/Wet)
    predicted_hidden_state = np.argmax(filtered_probs[i])
    
    # Get the actual weather category from the observation
    actual_obs = observations[i]
    actual_category = 1 if obs_labels[i] in ['Rainy', 'Snowy'] else 0  # Wet=1, Dry=0
    
    # Compare predicted category with actual category
    if predicted_hidden_state == actual_category:
        category_correct += 1

category_accuracy = category_correct / total_predictions if total_predictions > 0 else 0

# Method 2: Basic weather type prediction - 3 points
# THINK: How do you use hidden state probabilities to predict weather types?
obs_correct = 0
for i in range(total_predictions):
    # TODO: Get hidden state probabilities
    hidden_probs = filtered_probs[i]
    
    # TODO: Calculate weighted observation probabilities
    # THINK: How do you combine hidden state probabilities with emission probabilities?
    obs_probs = np.zeros(len(observation_symbols))
    for hidden_state in range(len(hidden_states)):
        obs_probs += hidden_probs[hidden_state] * emission_probs[hidden_state]
    
    # TODO: Predict most likely observation
    predicted_obs = np.argmax(obs_probs)
    
    # TODO: Compare with actual observation
    if predicted_obs == observations[i]:
        obs_correct += 1

obs_accuracy = obs_correct / total_predictions if total_predictions > 0 else 0

# Method 3: Enhanced weather type prediction - 4 points
# THINK: How can you improve weather type prediction beyond basic methods?
# THINK: What additional information could help predict weather types more accurately?
# THINK: How can you use the original temperature data to improve predictions?

# TODO: Calculate enhanced weather type accuracy
enhanced_obs_correct = 0
if smoothed_probs is not None:
    for i in range(min(len(observations), len(smoothed_probs))):
        # TODO: Implement enhanced prediction method
        # THINK: How can smoothed probabilities help?
        # THINK: How can temperature data from weather_df help?
        # THINK: What patterns exist between weather types and hidden states?
        pass

enhanced_obs_accuracy = enhanced_obs_correct / total_predictions if total_predictions > 0 else 0

# Print results
print(f"Total predictions made: {total_predictions}")
print(f"Correct predictions: {category_correct}")
print(f"Category accuracy (Dry/Wet): {category_accuracy:.3f}")
print(f"Weather type accuracy (Basic): {obs_accuracy:.3f}")
print(f"Weather type accuracy (Enhanced): {enhanced_obs_accuracy:.3f}")

# TODO: Add comprehensive documentation and print statements (5 points)
# THINK: What information would help someone understand your implementation?
print("\nData distribution:")
print(f"Dry days: {dry_count}")
print(f"Wet days: {wet_count}")
print(f"Weather type distribution: {dict(Counter(obs_labels))}")

# TODO: Add more documentation and verification prints
# THINK: What key insights about your HMM parameters and predictions should you share?
print("\nAll sections completed with comprehensive documentation:")
print("- Data loading and preprocessing: TODO")
print("- HMM parameter estimation: TODO")
print("- Forward algorithm implementation: TODO")
print("- Prediction implementation: TODO")
print("- Forward-backward algorithm implementation: TODO")
print("- Evaluation and accuracy calculation: TODO")
print("- Documentation and comments: TODO")

if __name__ == '__main__':
    print("HMM Weather Assignment: Complete each section and run this script.")
    print("Points in this file: 50")
    print("="*60)
    print("HMM WEATHER ANALYSIS COMPLETE")
    print("="*60) 