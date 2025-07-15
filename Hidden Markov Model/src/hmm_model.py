"""
Hidden Markov Model Implementation
Complete the TODO sections to implement the core HMM algorithms.

This module provides the HiddenMarkovModel class with methods for:
- Forward algorithm (filtering)
- Forward-backward algorithm (smoothing)  
- State prediction

POINTS IN THIS FILE: 50
"""

import numpy as np

class HiddenMarkovModel:
    """
    Hidden Markov Model for discrete weather states.
    
    Students must implement three core algorithms:
    1. Forward algorithm (filtering) - 25 points
    2. Forward-Backward algorithm (smoothing) - 20 points  
    3. State prediction - 5 points
    
    TOTAL POINTS IN THIS FILE: 50
    """
    def __init__(self, hidden_states, observation_symbols, initial_probs, transition_probs, emission_probs):
        """
        Initialize the HMM.
        Args:
            hidden_states (list): List of hidden state names.
            observation_symbols (list): List of observation symbol names.
            initial_probs (np.ndarray): Initial state probabilities.
            transition_probs (np.ndarray): State transition probability matrix.
            emission_probs (np.ndarray): Emission probability matrix.
        """
        self.hidden_states = hidden_states
        self.observation_symbols = observation_symbols
        self.initial_probs = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs

    def forward(self, observations):
        """
        Forward algorithm (filtering): Compute probability of each hidden state at each time step.
        
        FORMULA: alpha[t][i] = emission_probs[i][obs[t]] * sum(alpha[t-1][j] * transition_probs[j][i])
        
        Args:
            observations (list): Sequence of observed symbols (as indices).
        Returns:
            np.ndarray: Filtered probabilities (time_steps x num_states)
        
        POINTS: 25 total
        - Initialization: 10 points
        - Forward recursion: 15 points
        
        CONCEPTUAL GUIDANCE:
        - Think about how to initialize probabilities for the first time step
        - Consider how to combine emission and transition probabilities
        - Remember to use the forward recursion formula for subsequent steps
        """
        # TODO: Initialize alpha matrix (5 points)
        T = len(observations)
        N = len(self.hidden_states)
        alpha = np.zeros((T, N))
        
        # TODO: Initialize first time step (5 points)
        for i in range(N):
            # TODO: Set alpha[0][i] using initial probabilities and emission probabilities
            # THINK: How do you combine initial state probability with emission probability?
            pass
        
        # TODO: Forward recursion (15 points)
        for t in range(1, T):
            for i in range(N):
                # TODO: Calculate alpha[t][i] using forward recursion formula
                # THINK: You need to consider all possible previous states and their transitions
                # THINK: How do you combine emission probability with transition probability?
                pass
        
        return alpha

    def forward_backward(self, observations):
        """
        Forward-Backward algorithm (smoothing): Compute smoothed probabilities for each state at each time.
        
        FORMULA: 
        - Backward: beta[t][i] = sum(transition_probs[i][j] * emission_probs[j][obs[t+1]] * beta[t+1][j])
        - Smoothing: gamma[t][i] = alpha[t][i] * beta[t][i] / sum(alpha[t][k] * beta[t][k])
        
        Args:
            observations (list): Sequence of observed symbols (as indices).
        Returns:
            np.ndarray: Smoothed probabilities (time_steps x num_states)
        
        POINTS: 20 total
        - Forward call: 0 points (already implemented)
        - Backward initialization: 5 points
        - Backward recursion: 10 points
        - Smoothing combination: 5 points
        
        CONCEPTUAL GUIDANCE:
        - Start with the forward algorithm results
        - Think about how to initialize the backward probabilities at the end
        - Consider how to work backwards through the sequence
        - Remember to normalize when combining forward and backward results
        """
        # TODO: Get forward probabilities (0 points - already implemented)
        alpha = self.forward(observations)
        
        # TODO: Initialize backward probabilities (5 points)
        T = len(observations)
        N = len(self.hidden_states)
        beta = np.zeros((T, N))
        
        # TODO: Initialize last time step for backward algorithm (5 points)
        for i in range(N):
            # TODO: Set beta[T-1][i] to an appropriate value
            # THINK: What should the backward probability be at the final time step?
            pass
        
        # TODO: Backward recursion (10 points)
        for t in range(T-2, -1, -1):
            for i in range(N):
                # TODO: Calculate beta[t][i] using backward recursion formula
                # THINK: You need to consider all possible next states and their emissions
                # THINK: How do you combine transition, emission, and backward probabilities?
                pass
        
        # TODO: Combine forward and backward probabilities (5 points)
        smoothed = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                # TODO: Calculate smoothed[t][i] by combining alpha and beta
                # THINK: How do you combine forward and backward probabilities?
                pass
            # TODO: Normalize the smoothed probabilities for time t
            # THINK: Why is normalization important for probabilities?
            pass
        
        return smoothed

    def predict_next_state(self, last_filtered_probs):
        """
        Predict the next hidden state distribution given the last filtered probabilities.
        
        FORMULA: predicted[i] = sum(last_filtered_probs[j] * transition_probs[j][i])
        
        Args:
            last_filtered_probs (np.ndarray): Probability distribution over states at last time step.
        Returns:
            np.ndarray: Predicted state distribution for next time step.
        
        POINTS: 5 total
        - Prediction calculation: 5 points
        
        CONCEPTUAL GUIDANCE:
        - Think about how current state probabilities influence future states
        - Consider how transition probabilities affect predictions
        - Remember that you're predicting the distribution over all possible states
        """
        # TODO: Initialize prediction array (0 points)
        N = len(self.hidden_states)
        predicted = np.zeros(N)
        
        # TODO: Calculate prediction using transition probabilities (5 points)
        for i in range(N):
            # TODO: Calculate predicted[i] using the prediction formula
            # THINK: How do you use current state probabilities to predict future states?
            # THINK: What role do transition probabilities play in prediction?
            pass
        
        return predicted 