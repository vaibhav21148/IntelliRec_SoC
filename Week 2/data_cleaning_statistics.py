import pandas as pd

# Load datasets
movies = pd.read_csv("Week 2\\movies.csv")
ratings = pd.read_csv("Week 2\\ratings.csv")

# Merge datasets on movieId
df = pd.merge(ratings, movies, on='movieId')

# Preview
print("Merged Dataset Preview:")
print(df.head())

# Merged Dataset Preview:
#    userId  movieId  rating  timestamp                        title                                       genres
# 0       1        1     4.0  964982703             Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
# 1       1        3     4.0  964981247      Grumpier Old Men (1995)                               Comedy|Romance
# 2       1        6     4.0  964982224                  Heat (1995)                        Action|Crime|Thriller
# 3       1       47     5.0  964983815  Seven (a.k.a. Se7en) (1995)                             Mystery|Thriller
# 4       1       50     5.0  964982931   Usual Suspects, The (1995)                       Crime|Mystery|Thriller

# Column Names
print("\nColumn Names:")
print(df.columns)

# Column Names:
# Index(['userId', 'movieId', 'rating', 'timestamp', 'title', 'genres'], dtype='object')

# Missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Missing values per column:
# userId       0
# movieId      0
# rating       0
# timestamp    0
# title        0
# genres       0
# dtype: int64

# Drop missing values and duplicates
df_cleaned = df.dropna().drop_duplicates()

# Save cleaned version
df_cleaned.to_csv("Week 2\\cleaned_merged_movies.csv", index=False)
print("\nCleaned data saved to 'cleaned_merged_movies.csv'")
