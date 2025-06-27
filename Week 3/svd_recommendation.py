from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Load dataset
# Ensure 'userId', 'movieId', 'rating' columns exist
df = pd.read_csv("Week 3\\ratings.csv")  
print(df.head())

# Define Reader with rating scale
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Initialize and train SVD model
model = SVD()
model.fit(trainset)

# Predict on test set
predictions = model.test(testset)

# Evaluate performance
rmse = accuracy.rmse(predictions)
print(f"Test RMSE: {rmse:.4f}")
