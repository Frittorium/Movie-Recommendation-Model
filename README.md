# Movie Recommendation System

A personalized movie recommendation system built with PyTorch and the MovieLens 1M dataset. The project predicts user ratings and recommends movies based on user preferences, movie characteristics, and historical rating patterns.

## Overview

The **MovieLens 1M dataset** contains approximately one million ratings from 6,040 users across 3,883 movies. It includes user demographics, movie genres, and explicit ratings on a 1–5 scale.

The project addresses the challenge of helping users discover movies they are likely to enjoy by predicting their ratings and ranking unseen movies accordingly.

## Approach

I developed and evaluated multiple recommendation models, including matrix factorization, Ridge regression, gradient boosting, and a hybrid neural network.

The hybrid model was selected to combine collaborative filtering with user and movie metadata, capturing both latent preferences and additional contextual information.

## Implementation

The solution was implemented in Python using PyTorch, Pandas, NumPy, and scikit-learn.

Key steps included:

- **Data preprocessing:** Cleaned ratings, handled duplicates, encoded categorical features, and extracted movie metadata.
- **Feature engineering:** Incorporated user demographics, movie genres, release years, and title-based features.
- **Model development:** Combined user and movie embeddings, bias terms, and a multilayer perceptron to predict ratings.
- **Model evaluation:** Used an 80/10/10 training, validation, and test split to compare models using RMSE, MAE, and R².
- **Application:** Built a Tkinter desktop interface for personalized recommendations, movie search, and rating predictions.

## Results and Findings

The selected hybrid neural network achieved the following test-set results:

| Metric | Result |
|---|---:|
| RMSE | 0.8546 |
| MAE | 0.6679 |
| R² | 0.4140 |

The model combines collaborative filtering with metadata-based features and outperformed the tested matrix factorization, linear regression, and gradient boosting alternatives in the reported test results.

## Features

- Personalized top-N movie recommendations
- Genre-based filtering
- Predicted ratings for individual movies
- Movie search and rating history
- User profile information and recommendation context
