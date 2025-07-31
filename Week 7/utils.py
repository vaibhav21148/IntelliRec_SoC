
# Utility functions for loading and encoding the dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_encode_data(filepath):
    """
    Loads ratings.csv, encodes userId and movieId, and splits data.
    Returns train/test sets and the number of users and movies.
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

def get_encoders(filepath):
    """
    Load the dataset and return encoders for userId and movieId.
    """
    data = pd.read_csv(filepath)
    user_encoder = {id: idx for idx, id in enumerate(data['userId'].unique())}
    movie_encoder = {id: idx for idx, id in enumerate(data['movieId'].unique())}
    movie_decoder = {idx: id for id, idx in movie_encoder.items()}
    return user_encoder, movie_encoder, movie_decoder

def load_movies(filepath):
    """
    Loads movies.csv file for title lookup.
    """
    return pd.read_csv(filepath)
