from ethical_manifolds import EthicalManifold
from ethical_manifolds.utils.data_processing import load_data, prepare_data
import sys

def main():
    try:
        print("Starting the train and test process...")

        # Define ethical dimensions
        ethical_dimensions = ['fairness', 'utility', 'virtue']
        print(f"Ethical dimensions: {ethical_dimensions}")

        # Create EthicalManifold instance
        print("Creating EthicalManifold instance...")
        manifold = EthicalManifold(ethical_dimensions)

        # Load and prepare data
        print("Loading data...")
        df = load_data('sample_data.csv')
        print(f"Data loaded. Shape: {df.shape}")

        print("Preparing data...")
        X_train, y_train, X_test, y_test = prepare_data(df, 'text', ethical_dimensions)
        print(f"Data prepared. Training set size: {len(X_train)}")

        # Train the model
        print("Training the model...")
        manifold.train(X_train, y_train, epochs=5, batch_size=32)

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

        print("Train and test process completed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
