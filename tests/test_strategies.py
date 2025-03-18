import unittest
from src.strategies.moving_average import MovingAverageStrategy

class TestMovingAverageStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = MovingAverageStrategy()

    def test_execute(self):
        # Test the execute method with sample data
        sample_data = [1, 2, 3, 4, 5]
        result = self.strategy.execute(sample_data)
        self.assertIsNotNone(result)

    def test_evaluate(self):
        # Test the evaluate method with sample results
        sample_results = {'profit': 100, 'loss': 50}
        evaluation = self.strategy.evaluate(sample_results)
        self.assertIn('profit', evaluation)
        self.assertIn('loss', evaluation)

if __name__ == '__main__':
    unittest.main()