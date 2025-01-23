import unittest
from data_preprocessing import load_data

class TestDataPreprocessing(unittest.TestCase):
    def test_load_data(self):
        df = load_data("data/raw/sample.csv")
        self.assertTrue('Date' in df.columns)
        self.assertTrue('Close' in df.columns)

if __name__ == '__main__':
    unittest.main()
