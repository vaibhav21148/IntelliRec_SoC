import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("Week 2\\cleaned_merged_movies.csv")

def basic_statistics():
    print("\nSummary Statistics:")
    print(df.describe())

    # Summary Statistics:
    #               userId        movieId         rating     timestamp
    # count  100836.000000  100836.000000  100836.000000  1.008360e+05
    # mean      326.127564   19435.295718       3.501557  1.205946e+09
    # std       182.618491   35530.987199       1.042529  2.162610e+08
    # min         1.000000       1.000000       0.500000  8.281246e+08
    # 25%       177.000000    1199.000000       3.000000  1.019124e+09
    # 50%       325.000000    2991.000000       3.500000  1.186087e+09
    # 75%       477.000000    8122.000000       4.000000  1.435994e+09
    # max       610.000000  193609.000000       5.000000  1.537799e+09

def data_info():
    print("\nDataset Info:")
    print(df.info())

    # Dataset Info:
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 100836 entries, 0 to 100835
    # Data columns (total 6 columns):
    #  #   Column     Non-Null Count   Dtype
    # ---  ------     --------------   -----
    #  0   userId     100836 non-null  int64
    #  1   movieId    100836 non-null  int64
    #  2   rating     100836 non-null  float64
    #  3   timestamp  100836 non-null  int64
    #  4   title      100836 non-null  object
    #  5   genres     100836 non-null  object
    # dtypes: float64(1), int64(3), object(2)
    # memory usage: 4.6+ MB
    # None

# ----- UNIVARIATE ANALYSIS -----

def genre_distribution():
    plt.figure(figsize=(12,6))
    df['genres'].str.split('|').explode().value_counts().head(10).plot(kind='bar', color='skyblue')
    plt.title('Top 10 Genres')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def rating_distribution():
    plt.figure(figsize=(8,6))
    sns.histplot(df['rating'], bins=10, kde=True, color='salmon')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# ----- BIVARIATE ANALYSIS -----

def mean_rating_by_genre():
    exploded = df.copy()
    exploded['genres'] = exploded['genres'].str.split('|')
    exploded = exploded.explode('genres')
    plt.figure(figsize=(12,6))
    exploded.groupby('genres')['rating'].mean().sort_values(ascending=False).head(10).plot(kind='bar', color='mediumseagreen')
    plt.title('Top 10 Genres by Average Rating')
    plt.xlabel('Genre')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def correlation_heatmap():
    plt.figure(figsize=(8,6))
    sns.heatmap(df[['userId', 'movieId', 'rating']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# ---------- MENU SYSTEM ----------

def main():
    options = {
        "1": ("Basic Statistics", basic_statistics),
        "2": ("Data Information", data_info),
        "3": ("Genre Distribution", genre_distribution),
        "4": ("Rating Distribution", rating_distribution),
        "5": ("Mean Rating by Genre", mean_rating_by_genre),
        "6": ("Correlation Heatmap", correlation_heatmap),
    }

    print("Select an option to perform analysis:")
    for key, (desc, _) in options.items():
        print(f"{key}. {desc}")

    choice = input("Enter your choice (1-6): ").strip()

    if choice in options:
        print(f"\nYou selected: {options[choice][0]}")
        options[choice][1]()  # Call the function
    else:
        print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()