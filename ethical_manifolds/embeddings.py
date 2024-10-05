import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .models.embedding_model import EthicalEmbeddingModel

class EmbeddingManager:
    def __init__(self, vocab_size, embedding_dim, max_length):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = EthicalEmbeddingModel(vocab_size, embedding_dim, max_length)
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

    def fit_tokenizer(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def get_embedding(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return self.model.get_embedding(padded)[0]

    def train(self, texts, y, epochs=10, batch_size=32):
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        self.model.train(X, y, epochs=epochs, batch_size=batch_size)

    def save(self, filepath):
        self.model.save(filepath + "_model")
        import pickle
        with open(filepath + "_tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)

    @classmethod
    def load(cls, filepath):
        model = EthicalEmbeddingModel.load(filepath + "_model")
        import pickle
        with open(filepath + "_tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)
        instance = cls(model.vocab_size, model.embedding_dim, model.max_length)
        instance.model = model
        instance.tokenizer = tokenizer
        return instance

embedding_manager = EmbeddingManager(vocab_size=10000, embedding_dim=100, max_length=200)

def get_embedding(text):
    return embedding_manager.get_embedding(text)
