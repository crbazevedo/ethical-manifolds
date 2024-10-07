from .models.embedding_model import EthicalEmbeddingModel
import numpy as np
from tensorflow import keras
import tensorflow as tf

class EmbeddingManager:
    def __init__(self, vocab_size=10000, embedding_dim=100, max_length=200):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Create TextVectorization layer
        self.tokenizer = keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=max_length,
            output_mode='int'
        )
        
        # Create embedding layer
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        )

    def fit_tokenizer(self, texts):
        print("Fitting tokenizer on texts:", texts[:5])
        self.tokenizer.adapt(texts)

    def tokenize(self, text):
        return self.tokenizer([text])

    def embed(self, tokenized_texts):
        return self.embedding(tokenized_texts)

    def pad_sequences(self, sequences):
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_length, 
            padding='post', 
            truncating='post'
        )

# Initialize the embedding manager
embedding_manager = EmbeddingManager(vocab_size=10000, embedding_dim=100, max_length=200)

def get_embedding(text):
    tokenized = embedding_manager.tokenize(text)
    embedded = embedding_manager.embed(tokenized)
    return embedded.numpy()[0]  # Return the full sequence of embeddings
