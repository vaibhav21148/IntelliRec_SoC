
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Load the ratings dataset
ratings = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 6\data\ratings.csv")

# Step 1: Prepare user and movie ID mappings
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_id_map = {id: idx for idx, id in enumerate(user_ids)}
movie_id_map = {id: idx for idx, id in enumerate(movie_ids)}

# Replace original IDs with mapped integer IDs
ratings["user_id_encoded"] = ratings["userId"].map(user_id_map)
ratings["movie_id_encoded"] = ratings["movieId"].map(movie_id_map)

num_users = len(user_ids)
num_movies = len(movie_ids)

# Step 2: Split the dataset into training and testing sets
X = ratings[["user_id_encoded", "movie_id_encoded"]].values
y = ratings["rating"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define embedding sizes
embedding_size = 50

# User embedding model
user_input = Input(shape=(1,), name="user_input")
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name="user_embedding")(user_input)
user_vector = Flatten()(user_embedding)

# Movie embedding model
movie_input = Input(shape=(1,), name="movie_input")
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, name="movie_embedding")(movie_input)
movie_vector = Flatten()(movie_embedding)

# Concatenate user and movie embeddings
merged_vector = Concatenate()([user_vector, movie_vector])

# Add fully connected layers
x = Dense(128, activation="relu")(merged_vector)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="linear", name="rating_output")(x)

# Build and compile the model
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# Print model summary
model.summary()

# Train the model
model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate on the test set
test_loss, test_mae = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print(f"\nTest MAE: {test_mae:.4f}")
model.save("deeplearning_model.h5", include_optimizer=False)
