# cold_start_solution.py

import pandas as pd

# Load data
movies = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 5\movies.csv")
ratings = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 5\ratings.csv")

# Merge to get genre-wise popularity
movie_data = pd.merge(movies, ratings, on='movieId')

# Split genres into multiple columns
genre_dummies = movies['genres'].str.get_dummies('|')
movies = pd.concat([movies, genre_dummies], axis=1)

# Compute average rating for each movie
movie_avg_ratings = movie_data.groupby('movieId')['rating'].mean().reset_index()
movies = pd.merge(movies, movie_avg_ratings, on='movieId', how='left')

# --- Cold Start Recommendation ---
def recommend_for_new_user(top_n=3):
    genre_cols = genre_dummies.columns
    results = []

    for genre in genre_cols:
        top_movies = movies[movies[genre] == 1].sort_values(by='rating', ascending=False).head(top_n)
        results.append((genre, top_movies['title'].tolist()))

    return results

# Show recommendations
recommendations = recommend_for_new_user()
for genre, movie_list in recommendations:
    print(f"Top {len(movie_list)} {genre} movies for cold-start:")
    for title in movie_list:
        print(f" - {title}")
    print()
