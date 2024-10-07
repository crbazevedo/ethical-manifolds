import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class EthicalClassifierModel:
    def __init__(self, input_dim, max_length, num_classes):
        self.input_dim = input_dim
        self.max_length = max_length
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.max_length, self.input_dim))
        x = Conv1D(64, 5, activation='relu')(inputs)
        x = GlobalMaxPooling1D()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='sigmoid')(x)
        
 
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mae'])
        return model

    def predict(self, embedding):
        return self.model.predict(embedding)

    def train(self, X, y, epochs=50, batch_size=32):
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
            X, y,
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
        instance = cls(model.input_shape[2], model.input_shape[1], model.output_shape[1])
        instance.model = model
        return instance

    def get_embedding_model(self):
        # Create a new model that outputs the embeddings
        embedding_model = Model(inputs=self.model.input, 
                                outputs=self.model.get_layer('global_max_pooling1d').output)
        return embedding_model

    def get_embeddings(self, X):
        embedding_model = self.get_embedding_model()
        return embedding_model.predict(X)
