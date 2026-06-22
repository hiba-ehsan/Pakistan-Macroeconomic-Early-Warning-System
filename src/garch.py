import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


def fitgarch(series, vol='GARCH', p=1, q=1):
    #Fit GARCH(1,1) or EGARCH(1,1) to a series.
    #Returns fit object or None if convergence fails.
    
    clean = series.dropna() * 100  # scale for numerical stability
    try:
        mod = arch_model(clean, vol=vol, p=p, q=q, dist='normal')
        fit = mod.fit(disp='off', options={'maxiter': 500})
        return fit
    except Exception as e:
        print(f"  {vol} failed for {series.name}: {e}")
        return None


def allgarch(diff_df, series_labels):
    #fit GARCH(1,1) and EGARCH(1,1) for all four series.
    #returns nested dict: {col: {'GARCH': fit, 'EGARCH': fit}}
    
    results = {}
    for col, label in series_labels.items():
        print(f"Fitting GARCH models: {label}")
        s = diff_df[col]
        results[col] = {
            'GARCH':  fitgarch(s, vol='GARCH'),
            'EGARCH': fitgarch(s, vol='EGARCH'),
            'label':  label,
        }
    return results


def summary(garch_results):
    #AIC& BIC comparison table across all series and models
    rows = []
    for col, res in garch_results.items():
        row = {'Indicator': res['label']}
        for model_name in ['GARCH', 'EGARCH']:
            fit = res.get(model_name)
            if fit:
                aic_val = fit.aic
                bic_val = fit.bic
                # Flag clearly inadmissible fits (convergence failures)
                if aic_val > 1e6 or bic_val > 1e6:
                    row[f'{model_name} AIC'] = f'{aic_val:.0f} (failed)'
                    row[f'{model_name} BIC'] = f'{bic_val:.0f} (failed)'
                else:
                    row[f'{model_name} AIC'] = round(aic_val, 3)
                    row[f'{model_name} BIC'] = round(bic_val, 3)
            else:
                row[f'{model_name} AIC'] = 'N/A'
                row[f'{model_name} BIC'] = 'N/A'
        g = res.get('GARCH')
        e = res.get('EGARCH')
        # Only prefer EGARCH if both converged to admissible fits
        if g and e and e.aic < g.aic and e.aic < 1e6:
            row['Better Model'] = 'EGARCH'
        elif g:
            row['Better Model'] = 'GARCH'
        else:
            row['Better Model'] = 'N/A'
        rows.append(row)
    return pd.DataFrame(rows).set_index('Indicator')


def conditionalvol(garch_results, diff_df):
    #for each indicator

    vol_df = pd.DataFrame(index=diff_df.index)
    for col, res in garch_results.items():
        g = res.get('GARCH')
        e = res.get('EGARCH')
        fit = e if (e and g and e.aic < g.aic) else g
        if fit:
            cond_vol = fit.conditional_volatility
            idx = diff_df.dropna().index[:len(cond_vol)]
            s = pd.Series(cond_vol.values, index=idx, name=col)
            vol_df[col] = s
    return vol_df