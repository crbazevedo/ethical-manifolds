from .embeddings import embedding_manager, get_embedding
from .classifiers import ClassifierManager
from .visualization import visualize_embedding, plot_ethical_scores
import numpy as np

class EthicalManifold:
    def __init__(self, ethical_dimensions):
        self.ethical_dimensions = ethical_dimensions
        self.classifier_manager = ClassifierManager(embedding_manager.embedding_dim, embedding_manager.max_length, ethical_dimensions)

    def analyze(self, text):
        embedding = get_embedding(text)
        return self.classifier_manager.classify_embedding(embedding)

    def train(self, X, y, epochs=50, batch_size=32):
        print("X shape:", X.shape if hasattr(X, 'shape') else len(X))
        print("y shape:", y.shape)
        print("X dtype:", X.dtype if hasattr(X, 'dtype') else type(X[0]))
        print("First few X values:", X[:5])
        print("First few y values:", y[:5])
        
        # Ensure X and y have the same number of samples
        if len(X) != y.shape[0]:
            raise ValueError(f"Mismatch in number of samples: X has {len(X)}, y has {y.shape[0]}")
        
        # Fit tokenizer
        embedding_manager.fit_tokenizer(X)
        
        # Get embeddings
        embeddings = np.array([get_embedding(x) for x in X])
        print("Embeddings shape:", embeddings.shape)
        
        # Prepare y_dict
        y_dict = {dim: y[:, i] for i, dim in enumerate(self.ethical_dimensions)}
        
        # Train classifiers
        histories = self.classifier_manager.train(embeddings, y_dict, epochs=epochs, batch_size=batch_size)
        return histories

    def save(self, dirpath):
        embedding_manager.save(dirpath + "/embedding")
        self.classifier_manager.save(dirpath + "/classifiers")

    @classmethod
    def load(cls, dirpath, ethical_dimensions):
        global embedding_manager
        embedding_manager = EmbeddingManager.load(dirpath + "/embedding")
        instance = cls(ethical_dimensions)
        instance.classifier_manager = ClassifierManager.load(dirpath + "/classifiers", ethical_dimensions)
        return instance

    def visualize_manifold(self, texts, labels):
        embeddings = [get_embedding(text) for text in texts]
        visualize_embedding(embeddings, labels, self.ethical_dimensions)

    def visualize_scores(self, text):
        scores = self.analyze(text)
        plot_ethical_scores(scores)

    def get_embeddings(self, X):
        # First, get the text embeddings from the EmbeddingManager
        text_embeddings = self.embedding_manager.get_embeddings(X)
        
        # Then, get the learned embeddings from each classifier
        learned_embeddings = {}
        for dim, classifier in self.classifier_manager.classifiers.items():
            learned_embeddings[dim] = classifier.get_embeddings(text_embeddings)
        
        return learned_embeddings
