import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
ratings = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 4\ratings.csv')
movies = pd.read_csv(r'C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 4\movies.csv')

# Create binary label: like (1) if rating >= 4, else 0
ratings['liked'] = (ratings['rating'] >= 4).astype(int)

# Use movieId as feature for simplicity (better features come later)
X = ratings[['movieId']]
y = ratings['liked']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
