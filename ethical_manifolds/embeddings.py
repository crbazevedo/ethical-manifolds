import numpy as np
import fasttext.util
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ssl
import certifi
import os

class EmbeddingManager:
    def __init__(self, max_length=200, model_lang='en'):
        # Disable SSL verification (not recommended for production)
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Now download the model
        fasttext.util.download_model(model_lang, if_exists='ignore')
        self.ft_model = fasttext.load_model(f'cc.{model_lang}.300.bin')
        self.max_length = max_length
        self.embedding_dim = self.ft_model.get_dimension()

    def tokenize(self, text):
        # FastText works directly with text, so we just need to split into words
        return text.lower().split()

    def embed(self, tokens):
        # Generate embeddings for each token
        embeddings = [self.ft_model.get_word_vector(token) for token in tokens]
        # Pad or truncate to max_length
        if len(embeddings) > self.max_length:
            embeddings = embeddings[:self.max_length]
        elif len(embeddings) < self.max_length:
            padding = [np.zeros(self.embedding_dim) for _ in range(self.max_length - len(embeddings))]
            embeddings.extend(padding)
        return np.array(embeddings)

    def embed_texts(self, texts):
        return np.array([self.embed(self.tokenize(text)) for text in texts])

    @property
    def vocab_size(self):
        return len(self.ft_model.get_words())

    def save(self, dirpath):
        os.makedirs(dirpath, exist_ok=True)
        np.save(os.path.join(dirpath, "max_length.npy"), self.max_length)
        # We don't need to save the FastText model as it's pre-trained and can be reloaded

    @classmethod
    def load(cls, dirpath, model_lang='en'):
        max_length = np.load(os.path.join(dirpath, "max_length.npy"))
        return cls(max_length=max_length, model_lang=model_lang)
