from .embeddings import EmbeddingManager
from .classifiers import ClassifierManager
from .visualization import visualize_embedding, plot_ethical_scores
import numpy as np
import os
from sklearn.metrics import mean_absolute_error

class EthicalManifold:
    def __init__(self, ethical_dimensions, max_length=200, model_lang='en'):
        self.ethical_dimensions = ethical_dimensions
        self.embedding_manager = EmbeddingManager(max_length=max_length, model_lang=model_lang)
        self.classifier_manager = ClassifierManager(
            vocab_size=self.embedding_manager.vocab_size,
            base_embedding_dim=self.embedding_manager.embedding_dim,  # This will be 300
            additional_embedding_dim=64,  # You can adjust this value
            max_length=max_length,
            ethical_dimensions=ethical_dimensions
        )

    def train(self, X, y, epochs=50, batch_size=32):
        X_embedded = self.embedding_manager.embed_texts(X)
        
        # Convert y to a dictionary
        y_dict = {dim: y[:, i] for i, dim in enumerate(self.ethical_dimensions)}
        
        # Train classifiers
        histories = self.classifier_manager.train(X_embedded, y_dict, epochs=epochs, batch_size=batch_size)
        return histories

    def analyze(self, text):
        # Tokenize the text
        tokens = self.embedding_manager.tokenize(text)
        
        # Convert tokens to numerical format
        X_text = np.array([self.embedding_manager.ft_model.get_word_id(token) for token in tokens])
        X_text = X_text.reshape(1, -1)  # Reshape to (1, sequence_length)
        
        # Pad or truncate to max_length
        if X_text.shape[1] < self.embedding_manager.max_length:
            X_text = np.pad(X_text, ((0, 0), (0, self.embedding_manager.max_length - X_text.shape[1])))
        elif X_text.shape[1] > self.embedding_manager.max_length:
            X_text = X_text[:, :self.embedding_manager.max_length]
        
        # Generate base embeddings
        X_base_embedding = self.embedding_manager.embed(tokens)
        X_base_embedding = X_base_embedding.reshape(1, *X_base_embedding.shape)  # Add batch dimension
        
        # Classify
        predictions = self.classifier_manager.classify_embedding(X_text, X_base_embedding)
        
        # Return the first (and only) prediction
        return predictions[0]

    def evaluate(self, X_test, y_test):
        X_text = []
        X_base_embedding = []
        
        for text in X_test:
            tokens = self.embedding_manager.tokenize(text)
            
            # Convert tokens to numerical format
            x_text = np.array([self.embedding_manager.ft_model.get_word_id(token) for token in tokens])
            
            # Pad or truncate to max_length
            if len(x_text) < self.embedding_manager.max_length:
                x_text = np.pad(x_text, (0, self.embedding_manager.max_length - len(x_text)))
            elif len(x_text) > self.embedding_manager.max_length:
                x_text = x_text[:self.embedding_manager.max_length]
            
            X_text.append(x_text)
            
            # Generate base embeddings
            x_base_embedding = self.embedding_manager.embed(tokens)
            X_base_embedding.append(x_base_embedding)
        
        X_text = np.array(X_text)
        X_base_embedding = np.array(X_base_embedding)
        
        y_pred = self.classifier_manager.classify_embedding(X_text, X_base_embedding)
        
        mae_scores = {}
        for i, dim in enumerate(self.ethical_dimensions):
            y_true = y_test[:, i]
            y_pred_dim = np.array([pred[dim] for pred in y_pred])
            
            mae = mean_absolute_error(y_true, y_pred_dim)
            mae_scores[dim] = mae
        
        return mae_scores

    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        embedding_dir = os.path.join(dirpath, "embedding")
        os.makedirs(embedding_dir, exist_ok=True)
        self.embedding_manager.save(embedding_dir)
        self.classifier_manager.save(os.path.join(dirpath, "classifiers"))

    @classmethod
    def load(cls, dirpath, ethical_dimensions):
        embedding_manager = EmbeddingManager.load(os.path.join(dirpath, "embedding"))
        classifier_manager = ClassifierManager.load(os.path.join(dirpath, "classifiers"), ethical_dimensions)
        
        instance = cls(ethical_dimensions)
        instance.embedding_manager = embedding_manager
        instance.classifier_manager = classifier_manager
        return instance

    def visualize_manifold(self, texts, labels):
        X_text = np.array([self.embedding_manager.tokenize(text)[0] for text in texts])
        X_base_embedding = np.array([self.embedding_manager.embed(tokens) for tokens in X_text])
        embeddings = self.classifier_manager.get_embeddings(X_text, X_base_embedding)
        visualize_embedding(embeddings, labels, self.ethical_dimensions)

    def visualize_scores(self, text):
        scores = self.analyze(text)
        plot_ethical_scores(scores)

    def get_embeddings(self, X):
        X_text = np.array([self.embedding_manager.tokenize(x)[0] for x in X])
        X_base_embedding = np.array([self.embedding_manager.embed(tokens) for tokens in X_text])        
        learned_embeddings = {}
        for dim, classifier in self.classifier_manager.classifiers.items():
            learned_embeddings[dim] = classifier.get_embeddings(X_text, X_base_embedding)
        
        return learned_embeddings
