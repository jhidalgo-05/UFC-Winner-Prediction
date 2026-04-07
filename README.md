# UFC Fight Winner Prediction

# Overview

This project focuses on predicting the winner of UFC fights using machine learning classification models. The goal is to leverage fighter-specific features and historical fight statistics to generate probabilistic predictions for upcoming bouts.

A key component of this project is the use of exponential decay weighting, which prioritizes recent fights over older ones. This approach is designed to better capture a fighter’s current form and improve forward-looking predictions.


# Important Consideration

While exponential decay helps improve predictions for future fights, it introduces a tradeoff:

Older fights receive less importance
Model evaluation on historical data may become biased
Backtesting performance may not fully reflect real-world predictive power

As a result, evaluation metrics should be interpreted with caution.


# Objectives

Predict the winner of UFC fights using classification models
Incorporate time-based weighting via exponential decay
Compare multiple ML models and evaluate performance
Analyze feature importance and fighter attributes


# Data

The dataset includes:

Fighter statistics (e.g., striking accuracy, takedowns, reach)
Fight outcomes
Event dates (used for time-weighting)

data sources: 

Kaggle: "maksbasher/ufc-complete-dataset-all-events-1996-2024"


# Methodology
1. Data Preprocessing
Clean and normalize fighter statistics
Handle missing values
Encode categorical variables
Create fighter comparison features (e.g., differences between fighters)

2. Exponential Decay Weighting

Each fight is assigned a weight based on recency:

weight = exp(-λ * time_difference)
Recent fights → higher weight
Older fights → lower weight

This emphasizes current fighter performance trends.

3. Feature Engineering
Fighter vs fighter differentials
Win/loss streaks
Time since last fight

4. Models Used
Logistic Regression
Gradient Boosting (e.g., XGBoost)
KNN (K-Nearest Neighbor)

5. Evaluation
Accuracy
ROC-AUC

Note: Traditional backtesting may be biased due to time-weighting.
