import pandas as pd
import numpy as np


CRISIS_PERIODS = [
    ('2008-01-01', '2009-06-01', '2008 Crisis'),
    ('2013-01-01', '2014-06-01', '2013 Crisis'),
    ('2018-01-01', '2019-12-01', '2018-19 Crisis'),
    ('2022-01-01', '2023-12-01', '2022-23 Crisis'),
]

CRISIS_ONSETS = {
    '2008 Crisis':   '2008-01-01',
    '2013 Crisis':   '2013-01-01',
    '2018-19 Crisis':'2018-01-01',
    '2022-23 Crisis':'2022-01-01',
}


def monthlyframe(cpi, fx, ca, gdp):
# Harmonise all four series to a unified monthly df. CPI & FX = alr monthly so just re-index. CA balance = quarterly so forward-fill. GDP = annual sp broadcast across 12 months for each year
    start = '2005-01-01'
    end   = '2026-04-01'
    idx   = pd.date_range(start=start, end=end, freq='MS')

    #CPI 
    cpi_m = cpi.copy()
    cpi_m.index = cpi_m.index.to_period('M').to_timestamp()
    cpi_m = cpi_m.reindex(idx)

    #FX
    fx_m = fx.copy()
    fx_m.index = fx_m.index.to_period('M').to_timestamp()
    fx_m = fx_m.reindex(idx)

    #CA balance
    ca_m = ca.copy()
    ca_m.index = ca_m.index.to_period('M').to_timestamp()
    ca_m = ca_m.reindex(idx).ffill()

    #GDP
    gdp_m = pd.Series(index=idx, dtype=float, name='GDP Growth')
    for year, val in gdp.items():
        gdp_m.loc[gdp_m.index.year == year] = val

    df = pd.DataFrame({
        'cpi_yoy':    cpi_m,
        'fx_rate':    fx_m,
        'ca_balance': ca_m,
        'gdp_growth': gdp_m,
    }, index=idx).dropna()

    return df


def transformations(df):
#CPI=first difference of YoY rate. FX=log-difference, % depreciation. CA balance and GDP=   first difference. Returns differenced df ready for GARCH/Markov.
    diff = pd.DataFrame({
        'cpi_change':  df['cpi_yoy'].diff(),
        'fx_log_diff': np.log(df['fx_rate']).diff() * 100,  # scale to %
        'ca_change':   df['ca_balance'].diff(),
        'gdp_change':  df['gdp_growth'].diff(),
    }).dropna()

    return diff


SERIES_LABELS = {
    'cpi_change':  'CPI Inflation',
    'fx_log_diff': 'FX Rate (PKR/USD)',
    'ca_change':   'Current Account',
    'gdp_change':  'GDP Growth',
}