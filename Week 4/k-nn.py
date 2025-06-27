import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load user-item matrix
df = pd.read_csv("user_item_matrix.csv", index_col=0)

# Transpose for item-based similarity
movie_user_matrix = df.T

# Fit k-NN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_user_matrix)

# Pick a movie to find similar ones
movie_title = "Inception"  # Change this to any movie in your data

# Check if movie exists
if movie_title not in movie_user_matrix.index:
    print(f"Movie '{movie_title}' not found in dataset.")
else:
    # Get recommendations
    distances, indices = model_knn.kneighbors(
        movie_user_matrix.loc[movie_title].values.reshape(1, -1),
        n_neighbors=6
    )

    print(f"\n🎬 Movies similar to '{movie_title}':")
    for i in range(1, len(distances[0])):
        print(f"{i}. {movie_user_matrix.index[indices[0][i]]} (Distance: {distances[0][i]:.2f})")
