from .embeddings import embedding_manager, get_embedding
from .classifiers import ClassifierManager
from .visualization import visualize_embedding, plot_ethical_scores

class EthicalManifold:
    def __init__(self, ethical_dimensions):
        self.ethical_dimensions = ethical_dimensions
        self.classifier_manager = ClassifierManager(embedding_manager.embedding_dim, ethical_dimensions)

    def analyze(self, text):
        embedding = get_embedding(text)
        return self.classifier_manager.classify_embedding(embedding)

    def train(self, texts, labels, epochs=10, batch_size=32):
        # First, fit the tokenizer
        embedding_manager.fit_tokenizer(texts)

        # Then, get embeddings for all texts
        embeddings = [get_embedding(text) for text in texts]

        # Train the classifiers
        self.classifier_manager.train(embeddings, labels, epochs=epochs, batch_size=batch_size)

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
