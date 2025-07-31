
# Generate top-N movie recommendations for a given user using the trained deep learning model.

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from utils import load_and_encode_data

# Load data and model
ratings = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 6\data\ratings.csv")
movies = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 6\movies.csv")
X_train, X_test, y_train, y_test, num_users, num_movies = load_and_encode_data(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 6\data\ratings.csv")

# Rebuild user/movie encoders
user_encoder = {id: idx for idx, id in enumerate(ratings['userId'].unique())}
user_decoder = {idx: id for id, idx in user_encoder.items()}
movie_encoder = {id: idx for idx, id in enumerate(ratings['movieId'].unique())}
movie_decoder = {idx: id for id, idx in movie_encoder.items()}

# Load trained model
model = load_model(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 6\deeplearning_model.h5", compile=False)25

# Function to recommend movies for a given userId
def recommend_movies(user_id, top_n=10):
    if user_id not in user_encoder:
        print("User not found in training data.")
        return []

    user_idx = user_encoder[user_id]
    all_movie_ids = ratings['movieId'].unique()
    seen_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    unseen_movies = [m for m in all_movie_ids if m not in seen_movies]

    user_input = np.full(len(unseen_movies), user_idx)
    movie_input = np.array([movie_encoder[m] for m in unseen_movies])

    preds = model.predict([user_input, movie_input], verbose=0).flatten()
    top_indices = preds.argsort()[-top_n:][::-1]
    top_movie_ids = [unseen_movies[i] for i in top_indices]

    recommendations = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]
    return recommendations

# Example usage
if __name__ == "__main__":
    user_id = int(input("Enter userId to get recommendations: "))
    recs = recommend_movies(user_id, top_n=5)
    print("\nTop Recommendations:")
    print(recs.to_string(index=False))
