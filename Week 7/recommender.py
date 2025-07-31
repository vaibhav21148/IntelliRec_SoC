
# Loads model and returns movie recommendations for a given user using a trained deep model.

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load data
ratings = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 7\data\ratings.csv")
movies = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 7\data\movies.csv")

# Rebuild encoders for userId and movieId
user_encoder = {id: idx for idx, id in enumerate(ratings['userId'].unique())}
movie_encoder = {id: idx for idx, id in enumerate(ratings['movieId'].unique())}
movie_decoder = {idx: id for id, idx in movie_encoder.items()}

# Load trained model
model = load_model(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 7\models\deeplearning_model.h5", compile=False)

def recommend_movies(user_id, top_n=5):
    if user_id not in user_encoder:
        return pd.DataFrame(columns=['movieId', 'title'])

    user_idx = user_encoder[user_id]
    all_movie_ids = ratings['movieId'].unique()
    seen_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    unseen_movies = [mid for mid in all_movie_ids if mid not in seen_movies]

    user_input = np.full(len(unseen_movies), user_idx)
    movie_input = np.array([movie_encoder[mid] for mid in unseen_movies])

    predictions = model.predict([user_input, movie_input], verbose=0).flatten()
    top_indices = predictions.argsort()[-top_n:][::-1]
    top_movie_ids = [unseen_movies[i] for i in top_indices]

    return movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]
