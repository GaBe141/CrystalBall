import pandas as pd

from src.utils import fit_theta_method

s = pd.Series([1.0, 2.0, 3.0])
for opt in ['linear','mean','last','zero','unknown']:
    res = fit_theta_method(s, test_size=1, h=4, short_series_fallback=opt)
    print(f"\n{opt}:")
    print(f"warning= {res.get('warning')}")
    print(f"model= {res.get('model')}")
    print(f"forecast_values= {list(res['forecast'].values)}")