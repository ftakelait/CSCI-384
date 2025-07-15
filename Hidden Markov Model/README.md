# Hidden Markov Model (HMM) Programming Assignment

## Overview
This assignment implements a Hidden Markov Model to analyze and predict weather patterns using real weather data from Grand Forks, ND. Students will learn to implement core HMM algorithms and apply them to a practical weather prediction problem.

## Assignment Structure

### Files Provided to Students:
- `src/dataset_utils.py` - **Complete solution provided** 
- `src/hmm_model.py` - **Student implementation required** (50 points)
- `src/hmm_project.py` - **Student implementation required** (50 points)
- `data/grand_forks_daily_weather_2019_2024.csv` - Weather dataset
- `grader.py` - Automated grading script

### Point Distribution (100 total):
- **HMM Model Implementation**: 50 points
- **HMM Project Workflow**: 50 points

## Learning Objectives
1. Understand Hidden Markov Models and their applications
2. Implement core HMM algorithms (Forward, Forward-Backward, Prediction)
3. Apply HMMs to real-world weather prediction
4. Evaluate model performance and interpret results

## 🌦️ Grand Forks Daily Weather Dataset

This dataset provides **real-world daily weather data** collected at **Grand Forks International Airport** between **January 1, 2019 and December 31, 2024**. It is intended for use in a programming assignment where students implement a **Hidden Markov Model (HMM)** for weather prediction.

### 📍 Dataset Source

- **Source**: [NOAA Climate Data Online (CDO)](https://www.ncei.noaa.gov/cdo-web/)
- **Dataset**: GHCN (Global Historical Climatology Network) - Daily
- **Station ID**: `GHCND:USW00014914`
- **Station Name**: Grand Forks International Airport
- **Location Info**: Latitude, Longitude, Elevation included

### 📅 Temporal Coverage

- **Start Date**: 2019-01-01  
- **End Date**: 2024-12-31  
- **Frequency**: Daily
- **Total Days**: ~2,192 days of weather data

### 📁 Format and Units

- **File Format**: CSV (Comma-Separated Values)
- **Units**:  
  - Temperature: Fahrenheit (°F)  
  - Precipitation / Snowfall: Inches  
  - Wind Speed: Miles per Hour (mph)  
- **Missing Values**: Indicated by values like `9999`, `99.99`, or left blank

### 📌 Variables Included

| Column | Description | Unit |
|--------|-------------|------|
| `DATE` | Date of observation (YYYY-MM-DD) | — |
| `PRCP` | Daily precipitation total | Inches |
| `SNOW` | Daily snowfall total | Inches |
| `TMAX` | Maximum daily temperature | °F |
| `TMIN` | Minimum daily temperature | °F |
| `AWND` | Average daily wind speed | mph |
| `STATION_NAME` | (Optional) Name of the weather station | — |
| `GEOGRAPHIC_LOCATION` | (Optional) Latitude, Longitude, Elevation | — |

### ⚠️ Data Quality Notes

- **Missing or unavailable values** are marked with placeholders like `9999` or empty cells.
- If data flags are present, they may indicate trace values or errors. For this assignment, they may be safely ignored unless otherwise specified.

### 🧪 Preprocessing for HMM

To apply Hidden Markov Models, we converted raw numerical data into discrete **weather states**:

**Weather Discretization Rules**:
- **Rainy**: Precipitation > 0.01 inches
- **Snowy**: Snowfall > 0.1 inches  
- **Sunny**: Max temperature > 75°F and no precipitation
- **Cloudy**: Max temperature 40-75°F and no precipitation
- **Cold**: Max temperature ≤ 40°F

**Hidden States**:
- **Dry**: Sunny, Cloudy, Cold weather patterns
- **Wet**: Rainy, Snowy weather patterns

This preprocessing allows the HMM to learn patterns in weather transitions and make predictions about future weather states.

## Understanding the Two Accuracy Metrics

### 1. Category Accuracy (Dry/Wet) - Your Main Goal
**Definition**: Measures how well the model predicts the correct hidden state (Dry or Wet).

**Typical Value**: ~0.82 (82%) with a well-implemented HMM.

**Why This Matters**:
- This is the main target for a 2-state HMM, since the hidden states are what the model is designed to infer. If you achieve >0.75 category accuracy, you have implemented the HMM correctly and learned the core concept.

**What It Means**: Your HMM can successfully distinguish between dry weather patterns (Sunny, Cloudy, Cold) and wet weather patterns (Rainy, Snowy).

### 2. Weather Type Accuracy (Rainy/Snowy/Sunny/Cloudy/Cold) - Advanced Challenge
**Definition**: Measures how well the model predicts the exact observed weather type for each day.

**Typical Values**:
- Basic HMM (2 hidden states): 0.30–0.35 (30–35%)
- With advanced enhancements: up to ~0.62 (62%) using temperature/seasonal rules.

**Why This Is Harder**:
- This is a much harder task for a 2-state HMM, since the model is not designed to distinguish all 5 weather types perfectly. You need to implement methods to improve this accuracy.

**What It Means**: Your HMM can predict the specific weather type, not just whether it's dry or wet.

## How to Approach This Assignment

**Your main goal is to build an HMM that can accurately infer the hidden weather state (Dry or Wet) from the observed data.**

**If your model achieves at least 75% accuracy in predicting Dry/Wet, you have done well!**

**Predicting the exact weather type (Rainy, Snowy, etc.) is much harder with a 2-state HMM, but you should still try to improve this accuracy.**

**Think creatively about how to use all available information to improve weather type predictions.**

## Assignment Steps

### Step 1: HMM Model Implementation (`hmm_model.py` - 50 points)

Complete the TODO sections in the `HiddenMarkovModel` class:

#### Forward Algorithm (25 points)
- **Initialization (10 points)**: Set up alpha matrix and initialize first time step
- **Forward Recursion (15 points)**: Implement the forward recursion formula
- **Conceptual Guidance**: Think about how to combine emission and transition probabilities

#### Forward-Backward Algorithm (20 points)
- **Backward Initialization (5 points)**: Initialize beta matrix for last time step
- **Backward Recursion (10 points)**: Implement backward recursion formula
- **Smoothing Combination (5 points)**: Combine alpha and beta with normalization
- **Conceptual Guidance**: Consider how to work backwards through the sequence

#### State Prediction (5 points)
- **Prediction Calculation (5 points)**: Use transition probabilities to predict next state
- **Conceptual Guidance**: Think about how current state probabilities influence future states

### Step 2: HMM Project Workflow (`hmm_project.py` - 50 points)

Complete the complete HMM workflow:

#### Data Loading and Preprocessing (5 points)
- Load weather data using provided `dataset_utils` (3 points)
- Convert weather observations to appropriate format (2 points)

#### HMM Model Setup (15 points)
- Define hidden states (e.g., Dry/Wet) (2 points)
- Map observations to hidden states (2 points)
- Estimate initial probabilities from data (3 points)
- Calculate transition probability matrix (3 points)
- Calculate emission probability matrix (5 points)

#### Algorithm Implementation (15 points)
- **Filtering**: Run forward algorithm (5 points)
- **Prediction**: Predict next weather state (5 points)
- **Smoothing**: Run forward-backward algorithm (5 points)

#### Evaluation and Documentation (15 points)
- Calculate prediction accuracy (10 points)
- Add comprehensive documentation and print statements (5 points)

## Getting Started

1. **Examine the provided files**:
   - `dataset_utils.py` is complete and ready to use
   - `hmm_model.py` contains TODO sections for algorithm implementation
   - `hmm_project.py` contains TODO sections for the complete workflow

2. **Start with `hmm_model.py`**:
   - Implement the forward algorithm first
   - Then implement the backward algorithm
   - Finally implement state prediction

3. **Complete `hmm_project.py`**:
   - Follow the step-by-step TODO comments
   - Use the provided `dataset_utils` for data loading
   - Test each section as you complete it

4. **Test your implementation**:
   ```bash
   python grader.py
   ```

## Grading Criteria

### HMM Model (50 points)
- **Forward Algorithm**: 25 points
  - Initialization: 10 points
  - Forward recursion: 15 points
- **Forward-Backward Algorithm**: 20 points
  - Backward initialization: 5 points
  - Backward recursion: 10 points
  - Smoothing combination: 5 points
- **State Prediction**: 5 points

### HMM Project (50 points)
- **Data Loading**: 5 points
- **Model Setup**: 15 points
- **Algorithm Calls**: 15 points
- **Evaluation**: 10 points
  - Basic accuracy calculation: 3 points
  - Evaluation loop implementation: 3 points
  - **Weather Type Accuracy Performance**: 4 points
    - Outstanding (4/4 points): >= 0.6 accuracy
    - Good (3/4 points): >= 0.5 accuracy
    - Acceptable (2/4 points): >= 0.4 accuracy
    - Needs improvement (1/4 points): < 0.4 accuracy
- **Documentation**: 5 points

## Weather Type Accuracy Targets

Students are expected to implement weather type prediction methods to achieve the following accuracy targets:

- **Outstanding Performance (4/4 points)**: Weather type accuracy >= 0.6 (60%)
- **Good Performance (3/4 points)**: Weather type accuracy >= 0.5 (50%)
- **Acceptable Performance (2/4 points)**: Weather type accuracy >= 0.4 (40%)
- **Needs Improvement (1/4 points)**: Weather type accuracy < 0.4 (40%)

### Conceptual Guidance for Weather Type Prediction

The assignment provides step-by-step guidance for implementing multiple prediction methods:

1. **Basic Weighted Probabilities**: Use hidden state probabilities and emission probabilities
2. **Smoothing**: Use forward-backward algorithm results for better predictions
3. **Enhanced Prediction**: Consider additional information beyond the basic HMM

**Conceptual Approaches for Achieving >0.5 Weather Type Accuracy:**
- Think about how to use smoothed probabilities instead of filtered probabilities
- Consider what additional information from the original weather data could help
- Explore patterns between weather types and hidden states
- Consider how temperature data might influence weather type predictions
- Think about seasonal patterns and weather transitions

**Note**: With a 2-state HMM (Dry/Wet), achieving weather type accuracy above 60% is challenging but demonstrates excellent understanding of HMM principles and creative prediction methods.

## Submission Requirements

1. **Complete implementation** of all TODO sections
2. **Working code** that runs without errors
3. **Proper documentation** with comments and print statements
4. **Accurate results** with reasonable weather predictions

## Tips for Success

1. **Start early** - This assignment requires understanding of HMM theory
2. **Test incrementally** - Test each algorithm as you implement it
4. **Understand the math** - Review HMM formulas before implementation
5. **Check edge cases** - Handle probability normalization properly
6. **For weather type accuracy** - Think creatively about how to use all available information

## Expected Outcomes

Students will produce:
- A working HMM implementation for weather prediction
- Analysis of weather patterns and transitions
- Evaluation of prediction accuracy
- Understanding of HMM applications in real-world problems

## Support

- Use the provided `dataset_utils.py` for data preprocessing
- Refer to HMM theory resources for algorithm understanding
- Test with the grader frequently to ensure correctness
- Ask questions early if concepts are unclear

