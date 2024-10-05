import argparse
from .manifold import EthicalManifold
from .utils.data_processing import load_data, prepare_data
from .utils.metrics import calculate_metrics, print_metrics

def main():
    parser = argparse.ArgumentParser(description="Ethical Manifolds CLI")
    parser.add_argument('action', choices=['train', 'analyze', 'evaluate'], help="Action to perform")
    parser.add_argument('--data', help="Path to the data file")
    parser.add_argument('--model', help="Path to save/load the model")
    parser.add_argument('--text', help="Text to analyze")
    parser.add_argument('--dimensions', nargs='+', default=['fairness', 'utility', 'virtue'], help="Ethical dimensions to consider")

    args = parser.parse_args()

    if args.action == 'train':
        if not args.data or not args.model:
            parser.error("--data and --model are required for training")

        df = load_data(args.data)
        X_train, y_train, X_test, y_test = prepare_data(df, 'text', args.dimensions)

        manifold = EthicalManifold(args.dimensions)
        manifold.train(X_train, y_train)
        manifold.save(args.model)

        print("Model trained and saved successfully.")

    elif args.action == 'analyze':
        if not args.model or not args.text:
            parser.error("--model and --text are required for analysis")

        manifold = EthicalManifold.load(args.model, args.dimensions)
        result = manifold.analyze(args.text)

        print("Ethical analysis results:")
        for dimension, score in result.items():
            print(f"{dimension.capitalize()}: {score:.2f}")

        manifold.visualize_scores(args.text)

    elif args.action == 'evaluate':
        if not args.data or not args.model:
            parser.error("--data and --model are required for evaluation")

        df = load_data(args.data)
        X_train, y_train, X_test, y_test = prepare_data(df, 'text', args.dimensions)

        manifold = EthicalManifold.load(args.model, args.dimensions)
        
        predictions = {}
        for text in X_test:
            result = manifold.analyze(text)
            for dimension, score in result.items():
                if dimension not in predictions:
                    predictions[dimension] = []
                predictions[dimension].append(round(score))

        for dimension in args.dimensions:
            print(f"\nMetrics for {dimension}:")
            metrics = calculate_metrics(y_test[dimension], predictions[dimension])
            print_metrics(metrics)

if __name__ == "__main__":
    main()