# utils.py
# ---------------------------------------------
# Utility functions for dataset preparation,
# encoding, and batching.
# ---------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_encode_data(filepath):
    """
    Loads the ratings data and encodes userId and movieId to integers.
    Returns train-test splits and number of unique users/movies.
    """
    data = pd.read_csv(filepath)

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    data['user'] = user_encoder.fit_transform(data['userId'])
    data['movie'] = movie_encoder.fit_transform(data['movieId'])

    num_users = data['user'].nunique()
    num_movies = data['movie'].nunique()

    X = data[['user', 'movie']].values
    y = data['rating'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, num_users, num_movies
