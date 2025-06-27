from data_preprocess import preprocess_dataset
from cosine_similarity import cosine_similarity

# Constants
ITEM_COLUMN = 'title'
FEATURE_COLUMN = 'genres'

# Load dataset with error handling
try:
    df, feature_matrix, all_features = preprocess_dataset(
    r"C:\Users\vaibh\OneDrive\Desktop\IITB\SoC_WnCC\IntelliRec_SoC\Week 1\testdataset.csv"
    )
except FileNotFoundError:
    print("❌ Error: CSV file not found. Check the path and filename.")
    exit()
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit()

# Recommendation function
def recommend(item_name, top_n=5):
    # Case-insensitive search
    matches = df[df[ITEM_COLUMN].str.lower() == item_name.lower()]
    
    if matches.empty:
        print(f"Item '{item_name}' not found in the dataset.")
        return

    index = matches.index[0]
    target_vector = feature_matrix[index]

    similarities = []
    for i, vec in enumerate(feature_matrix):
        if i != index:
            sim = cosine_similarity(target_vector, vec)
            similarities.append((df.iloc[i][ITEM_COLUMN], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} recommendations for '{df.iloc[index][ITEM_COLUMN]}':\n")
    for name, score in similarities[:top_n]:
        print(f"{name}  (Similarity: {score:.3f})")

# User input
user_input = input("Enter the name of the item (e.g. movie) : ").strip()
recommend(user_input, top_n=7)
