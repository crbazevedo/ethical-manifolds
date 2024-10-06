import pandas as pd
import numpy as np

def create_sample_data(n_samples=1000):
    np.random.seed(42)
    
    texts = [
        "AI should be ethical and fair.",
        "Maximize utility in all decisions.",
        "Virtue is the most important ethical principle.",
        "We must consider the consequences of our actions.",
        "Rights and duties are fundamental to ethics.",
    ]
    
    data = []
    for _ in range(n_samples):
        text = np.random.choice(texts)
        fairness = np.random.uniform(0, 1)
        utility = np.random.uniform(0, 1)
        virtue = np.random.uniform(0, 1)
        data.append([text, fairness, utility, virtue])
    
    df = pd.DataFrame(data, columns=['text', 'fairness', 'utility', 'virtue'])
    df.to_csv('sample_data.csv', index=False)
    print("Sample data created and saved to 'sample_data.csv'")

if __name__ == "__main__":
    create_sample_data()
