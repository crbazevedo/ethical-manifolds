import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data(filename):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to the project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Construct the full path to the data file
    filepath = os.path.join(project_root, 'data', filename)
    df = pd.read_csv(filepath)
    print("Columns in the loaded data:", df.columns.tolist())
    return df

def preprocess_text(text):
    """Preprocess text data."""
    # Add any text preprocessing steps here (e.g., lowercasing, removing punctuation)
    return text.lower()

def prepare_data(df, text_column, ethical_dimensions, test_size=0.2, random_state=42):
    # Separate features (X) and labels (y)
    X = df[text_column].values
    y = df[ethical_dimensions].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, y_train, X_test, y_test

def process_text(text):
    # This is a placeholder for text processing steps
    # Depending on the model architecture, this might involve TF-IDF, word embeddings, etc.
    return text
