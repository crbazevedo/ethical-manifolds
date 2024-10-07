import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/sample_data.csv')

# Display basic information
print(df.info())

# Show summary statistics
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check unique values in categorical columns
print("\nEthical label distribution:")
print(df['ethical_label'].value_counts(normalize=True))

# Check if all scores are between 0 and 1
score_columns = ['fairness_score', 'utility_score', 'virtue_score']
for col in score_columns:
    print(f"\n{col} range:")
    print(df[col].min(), df[col].max())