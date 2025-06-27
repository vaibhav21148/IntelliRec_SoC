import pandas as pd

# Load your cleaned merged dataset
df = pd.read_csv("Week 4\\cleaned_merged_movies.csv")

# Create user-item interaction matrix (ratings as values)
interaction_matrix = df.pivot_table(index='userId', columns='title', values='rating')

# Fill missing ratings with 0 (or use np.nan if needed)
interaction_matrix = interaction_matrix.fillna(0)

# Save the interaction matrix
interaction_matrix.to_csv("user_item_matrix.csv")
print("✅ User-Item matrix saved as 'user_item_matrix.csv'")