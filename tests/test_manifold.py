import unittest
from ethical_manifolds import EthicalManifold
from ethical_manifolds.utils.data_processing import prepare_data
import pandas as pd

class TestEthicalManifold(unittest.TestCase):
    def setUp(self):
        self.ethical_dimensions = ['fairness', 'utility', 'virtue']
        self.manifold = EthicalManifold(self.ethical_dimensions)

    def test_analyze(self):
        text = "AI should be developed with careful consideration of its ethical implications."
        result = self.manifold.analyze(text)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(self.ethical_dimensions))
        for score in result.values():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_train_and_save_load(self):
        # Create a small dummy dataset
        data = {
            'text': [
                "AI should be ethical",
                "Maximize utility",
                "Be virtuous in all actions"
            ],
            'fairness': [0.8, 0.6, 0.7],
            'utility': [0.6, 0.9, 0.5],
            'virtue': [0.7, 0.5, 0.9]
        }
        df = pd.DataFrame(data)

        # Prepare data
        X_train, y_train, _, _ = prepare_data(df, 'text', self.ethical_dimensions)

        # Train the model
        self.manifold.train(X_train, y_train, epochs=2, batch_size=2)

        # Save the model
        self.manifold.save("./test_model")

        # Load the model
        loaded_manifold = EthicalManifold.load("./test_model", self.ethical_dimensions)

        # Test the loaded model
        text = "AI ethics is important"
        original_result = self.manifold.analyze(text)
        loaded_result = loaded_manifold.analyze(text)

        for dim in self.ethical_dimensions:
            self.assertAlmostEqual(original_result[dim], loaded_result[dim], places=5)

if __name__ == '__main__':
    unittest.main()
