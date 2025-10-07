import numpy as np
import pandas as pd

from src.utils import fit_theta_method

# Test cases for different series lengths and patterns
test_series = {
    'tiny': pd.Series([1.0, 2.0, 3.0]),
    'tiny_flat': pd.Series([5.0, 5.0, 5.0]),
    'tiny_down': pd.Series([3.0, 2.0, 1.0]),
    'borderline': pd.Series(np.arange(4) + 1.0),
}

print("\nTesting fallback behaviors on short series:\n")
for name, s in test_series.items():
    print(f"\nSeries: {name} (length={len(s)}, values={list(s.values)})")
    for opt in ['linear', 'mean', 'last', 'zero']:
        res = fit_theta_method(s, test_size=1, h=3, short_series_fallback=opt)
        fc = list(res['forecast'].values)
        print(f"{opt:6s}: forecast={fc}, model={res.get('model', {}).get('method', 'unknown')}")