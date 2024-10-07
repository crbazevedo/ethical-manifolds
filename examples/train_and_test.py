from ethical_manifolds import EthicalManifold
from ethical_manifolds.utils.data_processing import load_data, prepare_data
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_history(histories, save_path='training_history.png'):
    fig, axes = plt.subplots(len(histories), 2, figsize=(15, 5*len(histories)))
    
    for i, (dimension, history) in enumerate(histories.items()):
        # Plot loss
        axes[i, 0].plot(history.history['loss'], label='Training Loss')
        axes[i, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[i, 0].set_title(f'{dimension} - Loss')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()
        
        # Plot MAE
        axes[i, 1].plot(history.history['mae'], label='Training MAE')
        axes[i, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[i, 1].set_title(f'{dimension} - Mean Absolute Error')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('MAE')
        axes[i, 1].legend()
    
    plt.tight_layout()
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory
    
    print(f"Training history plot saved to {save_path}")

def main():
    try:
        print("Starting the train and test process...")

        # Define ethical dimensions
        ethical_dimensions = ['fairness_score', 'utility_score', 'virtue_score']  # Updated to match CSV columns
        print(f"Ethical dimensions: {ethical_dimensions}")

        # Create EthicalManifold instance
        print("Creating EthicalManifold instance...")
        manifold = EthicalManifold(ethical_dimensions)

        # Load and prepare data
        print("Loading data...")
        df = load_data('sample_data.csv')
        print(f"Data loaded. Shape: {df.shape}")
        print(f"Columns in the loaded data: {df.columns}")

        print("Preparing data...")
        X_train, y_train, X_test, y_test = prepare_data(df, 'text', ethical_dimensions)
        print(f"Data prepared. Training set shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")

        # Debug: Print first few rows of X_train and y_train
        print("First few X_train values:")
        print(X_train[:5])
        print("First few y_train values:")
        print(y_train[:5])

        # Check for data consistency
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Mismatch in number of samples: X_train has {X_train.shape[0]}, y_train has {y_train.shape[0]}")

        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"Mismatch in number of samples: X_test has {X_test.shape[0]}, y_test has {y_test.shape[0]}")

        # Convert X_train to strings if it's not already
        if not isinstance(X_train[0], str):
            X_train = [' '.join(map(str, row)) for row in X_train]
            X_test = [' '.join(map(str, row)) for row in X_test]
            print("Converted X_train and X_test to strings")

        # Train the model
        print("Training the model...")
        histories = manifold.train(X_train, y_train, epochs=50, batch_size=32)
        
        print("Plotting training history...")
        plot_training_history(histories, save_path='results/training_history.png')

        # Save the trained model
        print("Saving the model...")
        manifold.save('./trained_model')
        print("Model trained and saved.")

        # Load the trained model
        print("Loading the saved model...")
        loaded_manifold = EthicalManifold.load('./trained_model', ethical_dimensions)

        # Analyze some text
        test_texts = [
            "AI should be developed with careful consideration of its ethical implications.",
            "We should always act to maximize overall happiness.",
            "The right action is the one that follows moral rules.",
        ]

        for text in test_texts:
            print(f"\nAnalyzing: '{text}'")
            result = loaded_manifold.analyze(text)
            for dimension, score in result.items():
                print(f"{dimension.capitalize()}: {score:.2f}")

        # Evaluate the model
        print("Evaluating the model...")
        accuracies = manifold.evaluate(X_test, y_test)
        for dimension, accuracy in accuracies.items():
            print(f"Model accuracy for {dimension}: {accuracy:.2f}")

        print("Train and test process completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
