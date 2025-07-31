# compare_models.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import numpy as np

# Load datasets
ratings = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 4\ratings.csv")  # Update path if needed
movies = pd.read_csv(r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 4\movies.csv")

# ---- k-NN Model Setup ----
def create_user_item_matrix(ratings_df):
    return ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

def knn_recommend(user_item_matrix, user_id, k=5):
    user_vec = user_item_matrix.loc[user_id].values.reshape(1, -1)
    sim_scores = cosine_similarity(user_vec, user_item_matrix.values)[0]
    sim_users = np.argsort(sim_scores)[::-1][1:k+1]
    sim_user_ids = user_item_matrix.index[sim_users]
    return sim_user_ids

# ---- Logistic Regression Model Setup ----
def prepare_logistic_data(ratings_df):
    df = ratings_df.copy()
    df['like'] = (df['rating'] >= 4).astype(int)
    X = df[['userId', 'movieId']]
    y = df['like']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_logistic(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    X_train_enc = pd.get_dummies(X_train.astype(str))
    X_test_enc = pd.get_dummies(X_test.astype(str))
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    model.fit(X_train_enc, y_train)
    preds = model.predict(X_test_enc)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    return precision, recall

# ---- Evaluation Function ----
def compare_models():
    print("=== k-NN Evaluation ===")
    matrix = create_user_item_matrix(ratings)
    user_id = 1
    top_sim_users = knn_recommend(matrix, user_id)
    print(f"Top {len(top_sim_users)} similar users to user {user_id}:", list(top_sim_users))

    print("\n=== Logistic Regression Evaluation ===")
    X_train, X_test, y_train, y_test = prepare_logistic_data(ratings)
    precision, recall = evaluate_logistic(X_train, X_test, y_train, y_test)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

# Run comparison
compare_models()
