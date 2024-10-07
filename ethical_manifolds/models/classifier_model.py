import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class EthicalClassifierModel:
    def __init__(self, vocab_size, base_embedding_dim, additional_embedding_dim, max_length, num_classes):
        self.vocab_size = vocab_size
        self.base_embedding_dim = base_embedding_dim
        self.additional_embedding_dim = additional_embedding_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        input_text = Input(shape=(self.max_length,), dtype='int32', name='input_text')
        input_base_embedding = Input(shape=(self.max_length, self.base_embedding_dim), name='input_base_embedding')
        
        # Additional embedding layer specific to this dimension
        x = Embedding(self.vocab_size, self.additional_embedding_dim, input_length=self.max_length, name='additional_embedding')(input_text)
        
        # Concatenate base embeddings with dimension-specific embeddings
        x = Concatenate(axis=2)([input_base_embedding, x])
        
        # Adjust the number of filters or add more layers if needed
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='sigmoid')(x)
        
        model = Model(inputs=[input_text, input_base_embedding], outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
        return model

    def predict(self, X_text, X_base_embedding):
        return self.model.predict([X_text, X_base_embedding])

    def train(self, X_text, X_base_embedding, y, epochs=50, batch_size=32):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        history = self.model.fit(
            [X_text, X_base_embedding], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=2
        )
        return history

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        model = tf.keras.models.load_model(filepath)
        
        # Find the embedding layer
        embedding_layer = model.get_layer('additional_embedding')
        
        if embedding_layer is None:
            raise ValueError("No Embedding layer found in the loaded model")
        
        vocab_size = embedding_layer.input_dim
        additional_embedding_dim = embedding_layer.output_dim
        
        # Get base_embedding_dim and max_length from the input shape of the model
        input_shapes = [input.shape for input in model.inputs]
        base_embedding_dim = input_shapes[1][-1]  # Assuming the second input is base_embedding
        max_length = input_shapes[0][1]  # Assuming the first input is input_text
        
        num_classes = model.output_shape[1]
        
        instance = cls(vocab_size, base_embedding_dim, additional_embedding_dim, max_length, num_classes)
        instance.model = model
        return instance

    def get_embedding_model(self):
        # Create a new model that outputs the embeddings
        embedding_model = Model(
            inputs=self.model.inputs, 
            outputs=self.model.get_layer('global_max_pooling1d').output
        )
        return embedding_model

    def get_embeddings(self, X_text, X_base_embedding):
        embedding_model = self.get_embedding_model()
        return embedding_model.predict([X_text, X_base_embedding])
