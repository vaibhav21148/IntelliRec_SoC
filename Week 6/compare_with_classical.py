
# Compares the deep learning model against classical models using MAE and MSE as evaluation metrics.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model

# Load dataset
ratings = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 6\data\ratings.csv")

# Encode user and movie IDs
user_ids = ratings["userId"].astype("category").cat.codes
movie_ids = ratings["movieId"].astype("category").cat.codes
ratings["user"] = user_ids
ratings["movie"] = movie_ids

# Prepare features and target
X = ratings[["user", "movie"]].values
y = ratings["rating"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classical Model 1: KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

# Classical Model 2: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Deep Learning Model (retrained or loaded)
# Assume deep model is trained already using deep_model.py
# You can save and load it like this:
# model.save("deep_model.h5")
# model = load_model("deep_model.h5")

# For demonstration, retrain the same deep model quickly
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

n_users = ratings["user"].nunique()
n_movies = ratings["movie"].nunique()

user_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=n_users, output_dim=50)(user_input)
user_vec = Flatten()(user_embedding)

movie_input = Input(shape=(1,))
movie_embedding = Embedding(input_dim=n_movies, output_dim=50)(movie_input)
movie_vec = Flatten()(movie_embedding)

merged = Concatenate()([user_vec, movie_vec])
x = Dense(128, activation='relu')(merged)
x = Dense(64, activation='relu')(x)
output = Dense(1)(x)

model = Model([user_input, movie_input], output)
model.compile(loss='mse', optimizer=Adam(0.001))
model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=3, batch_size=64, verbose=0)
dnn_preds = model.predict([X_test[:, 0], X_test[:, 1]]).flatten()

# Evaluation
print("\nModel Comparison Results:")
print("------------------------")
print(f"KNN MAE: {mean_absolute_error(y_test, knn_preds):.4f}, MSE: {mean_squared_error(y_test, knn_preds):.4f}")
print(f"Linear Reg MAE: {mean_absolute_error(y_test, lr_preds):.4f}, MSE: {mean_squared_error(y_test, lr_preds):.4f}")
print(f"Deep Model MAE: {mean_absolute_error(y_test, dnn_preds):.4f}, MSE: {mean_squared_error(y_test, dnn_preds):.4f}")
