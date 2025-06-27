from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Load dataset
# Ensure it has userId, movieId, rating
df = pd.read_csv("Week 3\\ratings.csv")  

# Setup for Surprise library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train collaborative filtering model
model = SVD()
model.fit(trainset)

# Evaluate model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"\nModel RMSE on test data: {rmse:.4f}")

# Prepare movie lists for each user
all_movieIds = df['movieId'].unique()
user_rated = df.groupby('userId')['movieId'].apply(set)

# Predict rating for a specific user-movie pair
def predict_for_user(userId, movieId):
    return model.predict(userId, movieId).est

# Recommend top-N unseen movies for a user
def recommend_top_n(userId, n=5):
    seen = user_rated.get(userId, set())
    unseen = [mid for mid in all_movieIds if mid not in seen]
    
    predictions = [(mid, predict_for_user(userId, mid)) for mid in unseen]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return top_n

if __name__ == "__main__":
    uid = int(input("\nEnter User ID (integer): "))
    n = int(input("How many movie recommendations? "))

    print(f"\nTop {n} movie recommendations for User {uid}:")
    results = recommend_top_n(uid, n)
    for movieId, score in results:
        print(f"Movie ID: {movieId}, Predicted Rating: {score:.2f}")
