import unittest

from src import utils


class TestUtils(unittest.TestCase):
    def test_example_util(self):
        self.assertIsNone(utils.example_util())

    def test_clean_and_detect(self):
        import pandas as pd
        df = pd.DataFrame({
            ' Date ': ['2020-01-01', '2020-02-01', None],
            'CPI Value': [100.0, 101.5, None],
            'Unnamed: 0': [None, None, None],
        })
        cleaned = utils.clean_df(df)
        # columns normalized and unnamed dropped
        self.assertIn('date', cleaned.columns)
        self.assertIn('cpi_value', cleaned.columns)
        self.assertNotIn('unnamed: 0', cleaned.columns)

        time_col = utils.detect_time_column(cleaned)
        self.assertEqual(time_col, 'date')

        cpi_col = utils.detect_cpi_column(cleaned)
        self.assertEqual(cpi_col, 'cpi_value')

    def test_summarize_stats(self):
        import pandas as pd
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        s = utils.summarize_stats(df)
        self.assertEqual(s['shape'], (3, 2))
        self.assertIn('a', s['numeric_summary'])
        self.assertIn('b', s['non_numeric_counts'])

    def test_fit_arima_series(self):
        import numpy as np
        import pandas as pd
        # Create a simple AR(1) series: x_t = 0.6 x_{t-1} + eps
        rng = np.random.RandomState(0)
        n = 50
        eps = rng.normal(scale=1.0, size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.6 * x[t-1] + eps[t]
        s = pd.Series(x)
        res = utils.fit_arima_series(s, test_size=10, auto_order=True, max_p=2, max_d=1, max_q=2)
        self.assertIsNone(res.get('error'))
        self.assertIn('forecast', res)
        self.assertIsNotNone(res['model'])
        self.assertIsNotNone(res['forecast'])
        # If metrics are present, they should be finite numbers
        if res.get('mae') is not None:
            self.assertTrue(res['mae'] >= 0)
        if res.get('rmse') is not None:
            self.assertTrue(res['rmse'] >= 0)

    def test_exponential_smoothing(self):
        import numpy as np
        import pandas as pd
        rng = np.random.RandomState(1)
        x = np.linspace(10, 20, 30) + rng.normal(scale=0.5, size=30)
        s = pd.Series(x)
        res = utils.fit_exponential_smoothing(s, test_size=5, trend='add')
        self.assertIsNone(res.get('error'))
        self.assertIn('forecast', res)

    def test_theta_and_croston(self):
        import numpy as np
        import pandas as pd
        # theta test on trending data
        rng = np.random.RandomState(2)
        x = np.linspace(0, 5, 40) + rng.normal(scale=0.2, size=40)
        s = pd.Series(x)
        res_theta = utils.fit_theta_method(s, test_size=5)
        self.assertIsNone(res_theta.get('error'))
        self.assertIn('forecast', res_theta)

        # croston test on intermittent demand
        arr = np.zeros(50)
        arr[[5, 10, 20, 35]] = [1, 2, 1, 3]
        s2 = pd.Series(arr)
        res_croston = utils.fit_croston(s2, n_forecast=3)
        self.assertIsNone(res_croston.get('error'))
        self.assertIn('forecast', res_croston)

if __name__ == "__main__":
    unittest.main()
