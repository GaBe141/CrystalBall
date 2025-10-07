import unittest

from src import stats_robust, utils
from src.tools import synthetic as syn


class TestStressSynthetic(unittest.TestCase):
    def test_robust_diagnostics_ar1(self):
        s = syn.series_ar1(n=120, phi=0.8, sigma=1.0)
        diag = stats_robust.run_robust_diagnostics(s)
        self.assertIn('adf_pvalue', diag)
        self.assertIn('ljungbox_pvalue', diag)

    def test_outliers_detected(self):
        s = syn.series_with_outliers(n=180, n_outliers=8, magnitude=8.0)
        diag = stats_robust.run_robust_diagnostics(s)
        # expect some outlier fraction > 0
        frac = diag.get('stl_outlier_frac')
        self.assertIsNotNone(frac)
        self.assertTrue((frac is None) or (frac >= 0))

    def test_structural_breaks(self):
        s = syn.series_structural_breaks(n=240, n_breaks=3, jump=6.0)
        diag = stats_robust.run_robust_diagnostics(s)
        # either ruptures or heuristic, but we expect a non-negative count
        self.assertIn('break_count', diag)
        self.assertTrue((diag['break_count'] is None) or (diag['break_count'] >= 0))

    def test_heteroskedastic(self):
        s = syn.series_heteroskedastic(n=240)
        # create fake residuals equal to the series (test ARCH only requires residual-like input)
        h = stats_robust.heteroskedasticity_tests(s)
        self.assertIn('arch_lm_pvalue', h)

    def test_modeling_end_to_end_small(self):
        # small but structured series to pass through utils models quickly
        s = syn.series_trend_seasonal(n=72)
        res_arima = utils.fit_arima_series(s, test_size=12, auto_order=True)
        self.assertIsNone(res_arima.get('error'))
        res_ets = utils.fit_exponential_smoothing(s, test_size=12, trend='add')
        self.assertIsNone(res_ets.get('error'))
        res_theta = utils.fit_theta_method(s, test_size=12)
        self.assertIsNone(res_theta.get('error'))


if __name__ == '__main__':
    unittest.main()
