import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def adf_test(series):
    stat, p, _, _, crit, _ = adfuller(series.dropna())
    return {
        'ADF Statistic': round(stat, 4),
        'p-value':       round(p, 4),
        'Critical 5%':   round(crit['5%'], 4),
        'Result':        'Stationary' if p < 0.05 else 'Non-Stationary'
    }


def kpss_test(series):
    stat, p, _, crit = kpss(series.dropna(), regression='c', nlags='auto')
    return {
        'KPSS Statistic': round(stat, 4),
        'p-value':        round(p, 4),
        'Critical 5%':    round(crit['5%'], 4),
        'Result':         'Stationary' if p > 0.05 else 'Non-Stationary'
    }


def archlm_test(series, lags=12):
#must run BEFORE fitting GARCH.
#H0: No ARCH effects (no volatility clustering).
#Rejecting H0 (p < 0.05) = ARCH effects present = GARCH justified.
    
    clean = series.dropna()
    lm_stat, lm_p, f_stat, f_p = het_arch(clean, nlags=lags)
    return {
        'LM Statistic': round(lm_stat, 4),
        'p-value':      round(lm_p, 4),
        'Result':       'ARCH Effects Present == (GARCH justified)' if lm_p < 0.05
                        else 'No ARCH Effects =/ (GARCH may not be appropriate)'
    }


def ljungbox_test(residuals, lags=10):
    #test on squared standardised residuals, run after GARCH.
    #H0: No autocorrelation remaining.
    #fail to reject (p > 0.05) = GARCH model adequate.

    sq_resid = residuals.dropna() ** 2
    result = acorr_ljungbox(sq_resid, lags=[lags], return_df=True)
    p = result['lb_pvalue'].values[0]
    return {
        'Q-Statistic': round(result['lb_stat'].values[0], 4),
        'p-value':     round(p, 4),
        'Result':      'Model is Adequate (no remaining autocorrelation)' if p > 0.05
                       else 'Model is Misspecified (autocorrelation remains)'
    }


def fulldiagnostics(diff_df, series_labels):
    #ADF, KPSS, and ARCH LM for all series
    #summary DataFrame gets returned
    
    rows = []
    for col, label in series_labels.items():
        s = diff_df[col].dropna()
        adf  = adf_test(s)
        kp   = kpss_test(s)
        arch = archlm_test(s)
        rows.append({
            'Indicator':        label,
            'ADF p-value':      adf['p-value'],
            'ADF Result':       adf['Result'],
            'KPSS p-value':     kp['p-value'],
            'KPSS Result':      kp['Result'],
            'ARCH LM p-value':  arch['p-value'],
            'ARCH Result':      arch['Result'],
        })
    return pd.DataFrame(rows).set_index('Indicator')


def plotacfpacf(diff_df, series_labels, lags=24):
    #ACF + PACF plots for all series
    n = len(series_labels)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    fig.suptitle('ACF and PACF — Differenced Series', fontsize=13, fontweight='bold')

    colors = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad']

    for i, (col, label) in enumerate(series_labels.items()):
        s = diff_df[col].dropna()

        acf_vals  = acf(s, nlags=lags, fft=True)
        pacf_vals = pacf(s, nlags=lags)
        ci = 1.96 / np.sqrt(len(s))

        #ACF
        ax_acf = axes[i, 0]
        ax_acf.bar(range(len(acf_vals)), acf_vals, color=colors[i], alpha=0.7, width=0.4)
        ax_acf.axhline(ci,  color='red', linestyle='--', linewidth=1)
        ax_acf.axhline(-ci, color='red', linestyle='--', linewidth=1)
        ax_acf.axhline(0,   color='black', linewidth=0.8)
        ax_acf.set_title(f'{label} — ACF', fontsize=10)
        ax_acf.set_xlabel('Lag')

        #PACF
        ax_pacf = axes[i, 1]
        ax_pacf.bar(range(len(pacf_vals)), pacf_vals, color=colors[i], alpha=0.7, width=0.4)
        ax_pacf.axhline(ci,  color='red', linestyle='--', linewidth=1)
        ax_pacf.axhline(-ci, color='red', linestyle='--', linewidth=1)
        ax_pacf.axhline(0,   color='black', linewidth=0.8)
        ax_pacf.set_title(f'{label} — PACF', fontsize=10)
        ax_pacf.set_xlabel('Lag')

    plt.tight_layout()
    return fig