import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ethical_manifolds import EthicalManifold

def plot_embeddings(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Load your data here (replace with your actual data loading code)
    X = np.random.rand(100, 10)  # Replace with your actual input data
    y = np.random.rand(100, 3)   # Replace with your actual labels

    # Load the trained EthicalManifold
    ethical_dimensions = ['fairness', 'utility', 'virtue']
    manifold = EthicalManifold.load('path/to/saved/model', ethical_dimensions)

    # Get the learned embeddings
    learned_embeddings = manifold.get_embeddings(X)

    # Analyze and visualize the embeddings
    for dim, embeddings in learned_embeddings.items():
        plot_embeddings(embeddings, y[:, ethical_dimensions.index(dim)], f'{dim.capitalize()} Embeddings')

    # Print some statistics
    for dim, embeddings in learned_embeddings.items():
        print(f"\n{dim.capitalize()} Embeddings:")
        print(f"Shape: {embeddings.shape}")
        print(f"Mean: {np.mean(embeddings)}")
        print(f"Std: {np.std(embeddings)}")
        print(f"Min: {np.min(embeddings)}")
        print(f"Max: {np.max(embeddings)}")

if __name__ == "__main__":
    main()