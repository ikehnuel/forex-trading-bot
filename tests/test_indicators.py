import unittest
from src.indicators.technical_indicators import calculate_moving_average, calculate_rsi

class TestTechnicalIndicators(unittest.TestCase):

    def test_calculate_moving_average(self):
        data = [1, 2, 3, 4, 5]
        period = 3
        expected_result = [None, None, 2.0, 3.0, 4.0]  # Assuming a simple moving average
        result = calculate_moving_average(data, period)
        self.assertEqual(result, expected_result)

    def test_calculate_rsi(self):
        data = [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
        period = 14
        expected_result = None  # Assuming not enough data for RSI calculation
        result = calculate_rsi(data, period)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()