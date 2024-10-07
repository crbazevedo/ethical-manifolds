from .models.classifier_model import EthicalClassifierModel

class ClassifierManager:
    def __init__(self, input_dim, max_length, ethical_dimensions):
        self.classifiers = {
            dimension: EthicalClassifierModel(input_dim=input_dim, max_length=max_length, num_classes=1)
            for dimension in ethical_dimensions
        }

    def classify_embedding(self, embedding):
        return {
            dimension: classifier.predict(embedding.reshape(1, embedding.shape[0], embedding.shape[1]))[0][0]
            for dimension, classifier in self.classifiers.items()
        }

    def train(self, X, y_dict, epochs=50, batch_size=32):
        histories = {}
        for dimension, classifier in self.classifiers.items():
            print(f"Training classifier for {dimension}")
            history = classifier.train(X, y_dict[dimension], epochs=epochs, batch_size=batch_size)
            histories[dimension] = history
        return histories

    def save(self, dirpath):
        for dimension, classifier in self.classifiers.items():
            classifier.save(f"{dirpath}/{dimension}_classifier.h5")

    @classmethod
    def load(cls, dirpath, ethical_dimensions):
        instance = cls(None, None, ethical_dimensions)  # We'll set input_dim and max_length later
        for dimension in ethical_dimensions:
            instance.classifiers[dimension] = EthicalClassifierModel.load(f"{dirpath}/{dimension}_classifier.h5")
        return instance

def classify_embedding(embedding):
    # This function should be implemented to use the ClassifierManager
    # You might want to have a global instance of ClassifierManager
    pass
