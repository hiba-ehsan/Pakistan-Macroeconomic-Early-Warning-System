import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import warnings
warnings.filterwarnings('ignore')


CRISIS_ONSETS = {
    '2008 Crisis':    pd.Timestamp('2008-01-01'),
    '2013 Crisis':    pd.Timestamp('2013-01-01'),
    '2018-19 Crisis': pd.Timestamp('2018-01-01'),
    '2022-23 Crisis': pd.Timestamp('2022-01-01'),
}


def markov(series):
    #Fit 2-state Markov Regime-Switching model.
    #Tries MarkovRegression first then falls back to MarkovAutoregression.

    clean = series.dropna()
    try:
        mod = MarkovRegression(
            clean, k_regimes=2, trend='c', switching_variance=True
        )
        return mod.fit(search_reps=20, disp=False)
    except Exception:
        try:
            mod = MarkovAutoregression(
                clean, k_regimes=2, order=1, switching_ar=False
            )
            return mod.fit(search_reps=20, disp=False)
        except Exception as e:
            print(f"  Markov failed for {series.name}: {e}")
            return None


def allmarkov(diff_df, series_labels):
    #for all 4 series
    results = {}
    for col, label in series_labels.items():
        print(f"Fitting Markov model: {label}")
        results[col] = {
            'fit':   markov(diff_df[col]),
            'label': label,
        }
    return results


def high_vol_regime(fit):
    """Return the regime index (0 or 1) with the larger unconditional variance."""
    try:
        params = fit.params
        if "sigma2[0]" in params.index and "sigma2[1]" in params.index:
            return 1 if params["sigma2[1]"] > params["sigma2[0]"] else 0
    except Exception:
        pass
    return 1


def regimeprobs(markov_results, diff_df):
    #smoothed high-volatility regime probabilities extraction
    #returns df with 1 column per indicator.
    
    prob_df = pd.DataFrame(index=diff_df.dropna().index)
    for col, res in markov_results.items():
        fit = res.get('fit')
        if fit is None:
            continue
        try:
            probs = fit.smoothed_marginal_probabilities
            years = diff_df.dropna().index
            n = min(len(probs), len(years))
            high_vol_col = high_vol_regime(fit)
            high_vol = probs.iloc[:n, high_vol_col]
            prob_df[col] = pd.Series(
                high_vol.values,
                index=years[:n],
                name=col
            )
        except Exception as e:
            print(f"  Could not extract probs for {col}: {e}")
    return prob_df


def filtered_regimeprobs(markov_results, diff_df):
    #filtered high-volatility regime probabilities extraction
    #returns df with 1 column per indicator.
    
    prob_df = pd.DataFrame(index=diff_df.dropna().index)
    for col, res in markov_results.items():
        fit = res.get('fit')
        if fit is None:
            continue
        try:
            probs = fit.filtered_marginal_probabilities
            years = diff_df.dropna().index
            n = min(len(probs), len(years))
            high_vol_col = high_vol_regime(fit)
            high_vol = probs.iloc[:n, high_vol_col]
            prob_df[col] = pd.Series(
                high_vol.values,
                index=years[:n],
                name=col
            )
        except Exception as e:
            print(f"  Could not extract filtered probs for {col}: {e}")
    return prob_df


def leadtimes(prob_df, series_labels, threshold=0.5, lookback_months=24):
    #For each indicator * crisis, find how many months before crisis onset the indicator first exceeded the selected threshold.
    
    records = []
    for col, label in series_labels.items():
        if col not in prob_df.columns:
            continue
        s = prob_df[col].dropna()
        for crisis_name, onset in CRISIS_ONSETS.items():
            window_start = onset - pd.DateOffset(months=lookback_months)
            window = s[(s.index >= window_start) & (s.index < onset)]
            above = window[window > threshold]
            if len(above) > 0:
                entry      = above.index[0]
                lead_months = (onset.to_period('M') - entry.to_period('M')).n
            else:
                entry       = None
                lead_months = None
            records.append({
                'Indicator':        label,
                'Crisis':           crisis_name,
                'Crisis Onset':     onset.strftime('%b %Y'),
                'Regime Entry':     entry.strftime('%b %Y') if entry else 'Not detected',
                'Lead Time (months)': lead_months,
            })
    return pd.DataFrame(records)