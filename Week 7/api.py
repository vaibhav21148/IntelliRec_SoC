
#Backend using Flask

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.losses import MeanSquaredError
import datetime


# Load data
ratings_df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 7\data\ratings.csv")
movies_df = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 7\data\movies.csv")
user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
# print("Valid User IDs:", user_item_matrix.index.tolist())

# Load model (replace with your trained model file)
model = load_model(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 7\models\deeplearning_model.h5", compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())


# Create mappings
user_map = {uid: idx for idx, uid in enumerate(ratings_df['userId'].unique())}
movie_map = {mid: idx for idx, mid in enumerate(ratings_df['movieId'].unique())}
reverse_movie_map = {idx: mid for mid, idx in movie_map.items()}
movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))

    if user_id not in user_map:
        return jsonify({'error': 'User ID not found'})

    user_idx = user_map[user_id]

    seen_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
    all_movies = np.array(list(movie_map.keys()))
    unseen_movies = np.setdiff1d(all_movies, seen_movies)

    user_indices = np.full_like(unseen_movies, user_idx)
    movie_indices = [movie_map[mid] for mid in unseen_movies]

    user_indices = np.array(user_indices, dtype=np.int32).reshape(-1, 1)
    movie_indices = np.array(movie_indices, dtype=np.int32).reshape(-1, 1)

    predictions = model.predict([user_indices, movie_indices], verbose=0).flatten()
    top_indices = predictions.argsort()[-5:][::-1]
    top_movie_ids = unseen_movies[top_indices]
    top_titles = [movie_id_to_title.get(mid, f"Movie {mid}") for mid in top_movie_ids]

    return jsonify({'user_id': user_id, 'recommendations': top_titles})

@app.route("/rating", methods=["POST"])
def save_rating():
    try:
        data = request.get_json()
        rating = data.get("rating")
        user_id = data.get("user_id")

        if rating is None or user_id is None:
            return jsonify({"error": "Missing rating or user_id"}), 400

        # Timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format entry
        entry = f"[{timestamp}] UserID: {user_id} ⭐ Rating: {rating}/5\n"

        # Append to rating.txt
        with open("rating.txt", "a", encoding="utf-8") as f:
            f.write(entry)

        print("Rating saved:", entry.strip())
        return jsonify({"message": "Rating saved"}), 200

    except Exception as e:
        print("Error saving rating:", str(e))
        return jsonify({"error": "Server error"}), 500



if __name__ == '__main__':
    app.run(debug=True)
