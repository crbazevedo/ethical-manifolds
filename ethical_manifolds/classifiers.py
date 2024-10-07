from .models.classifier_model import EthicalClassifierModel
import numpy as np

class ClassifierManager:
    def __init__(self, vocab_size, base_embedding_dim, additional_embedding_dim, max_length, ethical_dimensions):
        self.classifiers = {
            dimension: EthicalClassifierModel(
                vocab_size=vocab_size,
                base_embedding_dim=base_embedding_dim,
                additional_embedding_dim=additional_embedding_dim,
                max_length=max_length,
                num_classes=1
            )
            for dimension in ethical_dimensions
        }
        self.max_length = max_length

    def classify_embedding(self, X_text, X_base_embedding):
        predictions = []
        for i in range(len(X_text)):
            pred = {
                dimension: classifier.predict(X_text[i:i+1], X_base_embedding[i:i+1])[0][0]
                for dimension, classifier in self.classifiers.items()
            }
            predictions.append(pred)
        return predictions

    def train(self, X_embedded, y_dict, epochs=50, batch_size=32):
        histories = {}
        for dimension, classifier in self.classifiers.items():
            print(f"Training classifier for {dimension}")
            y = y_dict[dimension]
            X_text = np.arange(self.max_length).reshape(1, -1).repeat(len(X_embedded), axis=0)
            history = classifier.train(X_text, X_embedded, y, epochs=epochs, batch_size=batch_size)
            histories[dimension] = history
        return histories

    def save(self, dirpath):
        for dimension, classifier in self.classifiers.items():
            classifier.save(f"{dirpath}/{dimension}_classifier.h5")

    @classmethod
    def load(cls, dirpath, ethical_dimensions):
        first_model = EthicalClassifierModel.load(f"{dirpath}/{ethical_dimensions[0]}_classifier.h5")
        instance = cls(
            first_model.vocab_size,
            first_model.base_embedding_dim,
            first_model.additional_embedding_dim,
            first_model.max_length,
            ethical_dimensions
        )
        for dimension in ethical_dimensions:
            instance.classifiers[dimension] = EthicalClassifierModel.load(f"{dirpath}/{dimension}_classifier.h5")
        return instance

def classify_embedding(X_text, X_base_embedding):
    # This function should be implemented to use the ClassifierManager
    # You might want to have a global instance of ClassifierManager
    pass
