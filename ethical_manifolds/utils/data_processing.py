import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_text(text):
    """Preprocess text data."""
    # Add any text preprocessing steps here (e.g., lowercasing, removing punctuation)
    return text.lower()

def prepare_data(df, text_column, label_columns, test_size=0.2, random_state=42):
    """Prepare data for training and testing."""
    # Preprocess text
    df[text_column] = df[text_column].apply(preprocess_text)

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Prepare X and y for training
    X_train = train_df[text_column].tolist()
    y_train = {col: train_df[col].values for col in label_columns}

    # Prepare X and y for testing
    X_test = test_df[text_column].tolist()
    y_test = {col: test_df[col].values for col in label_columns}

    return X_train, y_train, X_test, y_test
