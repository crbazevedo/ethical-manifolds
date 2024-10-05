import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embedding(embeddings, labels, ethical_dimensions):
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Ethical Manifold Visualization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    # Add legend
    unique_labels = set(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=ethical_dimensions[label])
    plt.legend()

    plt.show()

def plot_ethical_scores(scores):
    dimensions = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(dimensions, values)
    plt.title('Ethical Dimension Scores')
    plt.xlabel('Ethical Dimensions')
    plt.ylabel('Score')
    plt.ylim(0, 1)

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')

    plt.show()
