import os

import pandas as pd

from src.api import DummyProvider, fetch_and_write_csv, init_default_providers, registry


def test_dummy_provider_fetch_shape_and_columns():
    p = DummyProvider()
    df = p.fetch_series('lorem-ipsum')
    assert isinstance(df, pd.DataFrame)
    assert set(['date', 'value', 'series', 'source']).issubset(df.columns)
    assert len(df) == 100
    assert df['series'].iloc[0] == 'lorem-ipsum'


def test_registry_and_csv_write(tmp_path):
    init_default_providers()
    assert 'dummy' in registry.names()
    out = fetch_and_write_csv('dummy', 'alpha-series', str(tmp_path))
    assert os.path.exists(out)
    df = pd.read_csv(out)
    assert 'date' in df.columns and 'value' in df.columns
