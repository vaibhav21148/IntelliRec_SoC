import numpy as np

def precision_at_k(actual, predicted, k):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(actual)) / k

def recall_at_k(actual, predicted, k):
    pred_k = predicted[:k]
    return len(set(pred_k) & set(actual)) / len(actual)

# Example usage
actual_movies = [1, 3, 7]
predicted_movies = [3, 7, 10, 20]

k = 3
print(f"Precision@{k}: {precision_at_k(actual_movies, predicted_movies, k):.2f}")
print(f"Recall@{k}: {recall_at_k(actual_movies, predicted_movies, k):.2f}")
