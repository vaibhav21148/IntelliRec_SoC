
import numpy as np

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
movies = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 5\movies.csv")
ratings = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 5\ratings.csv")

# --- CONTENT-BASED SIMILARITY ---
# TF-IDF on genres
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))

# Compute cosine similarity between movies
content_sim = cosine_similarity(tfidf_matrix)

# --- COLLABORATIVE FILTERING BASED SIMILARITY ---
# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute similarity between movies based on ratings
collab_sim = cosine_similarity(user_item_matrix.T)

# --- HYBRID SIMILARITY ---
# Ensure both similarity matrices have the same movie ordering
common_movie_ids = list(set(movies['movieId']).intersection(user_item_matrix.columns))
movies_index = movies[movies['movieId'].isin(common_movie_ids)].reset_index(drop=True)

common_movie_mask = movies['movieId'].isin(common_movie_ids).values
common_indices = np.where(common_movie_mask)[0]
content_sim = cosine_similarity(tfidf_matrix[common_indices])

# Final hybrid similarity = weighted average (0.5 for each)
hybrid_sim = 0.5 * content_sim + 0.5 * collab_sim

# --- Recommendation Function ---
def recommend_movies(movie_index, top_n=5):
    sim_scores = hybrid_sim[movie_index]
    top_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return movies_index.iloc[top_indices][['movieId', 'title']]

# Example: Recommend similar to movie at index 10
print("Recommendations based on hybrid model:")
print(recommend_movies(movie_index=10))
