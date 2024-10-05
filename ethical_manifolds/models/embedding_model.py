import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, LSTM, Bidirectional
from tensorflow.keras.models import Model

class EthicalEmbeddingModel:
    def __init__(self, vocab_size, embedding_dim, max_length, embedding_type='simple'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.embedding_type = embedding_type
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.max_length,))
        x = Embedding(self.vocab_size, self.embedding_dim)(inputs)

        if self.embedding_type == 'simple':
            x = GlobalAveragePooling1D()(x)
        elif self.embedding_type == 'lstm':
            x = LSTM(self.embedding_dim)(x)
        elif self.embedding_type == 'bilstm':
            x = Bidirectional(LSTM(self.embedding_dim // 2))(x)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

        outputs = Dense(self.embedding_dim, activation='relu')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_embedding(self, text):
        # Assume text is already tokenized and converted to sequence of integers
        return self.model.predict(text)

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        model = tf.keras.models.load_model(filepath)
        instance = cls(model.input_shape[1], model.output_shape[1], model.input_shape[1])
        instance.model = model
        return instance
