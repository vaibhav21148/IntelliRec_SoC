import pandas as pd

ITEM_COLUMN = 'title'
FEATURE_COLUMN = 'genres'

def preprocess_dataset(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataset Loaded:", df.shape)

    # Drop missing or duplicate entries
    df = df.dropna(subset=[ITEM_COLUMN, FEATURE_COLUMN])
    df = df.drop_duplicates(subset=[ITEM_COLUMN])
    
    # Normalize genres
    df[FEATURE_COLUMN] = df[FEATURE_COLUMN].str.lower().str.replace('|', ' ', regex=False)

    # One-hot encoding
    all_features = set()
    for row in df[FEATURE_COLUMN]:
        if isinstance(row, str):
            all_features.update(row.split())
    all_features = sorted(list(all_features))

    for feature in all_features:
        df[feature] = df[FEATURE_COLUMN].apply(
            lambda x: 1 if isinstance(x, str) and feature in x.split() else 0
    )

    feature_matrix = df[all_features].values.tolist()
    return df, feature_matrix, all_features
