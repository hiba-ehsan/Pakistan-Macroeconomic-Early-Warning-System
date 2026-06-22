import pandas as pd
import numpy as np
from pathlib import Path

#file's loc
_SRC_DIR  = Path(__file__).resolve().parent
DATA_DIR  = _SRC_DIR.parent / 'data'

CPI_PATH = DATA_DIR / 'pakistan_cpi_monthly_2001_2026.csv'
FX_PATH  = DATA_DIR / 'PKR_USD_Monthly_2005_2026.csv'
CA_PATH  = DATA_DIR / '_Current_Account_Balance__Summary_of_Balance_of_Payments_as_per_BPM6.csv'
GDP_PATH = DATA_DIR / 'Gross_Domestic_Product_of_Pakistan_at_constant_basic_prices_of_2015-16.csv'


def cpi():
    #Monthly CPI yoy inflation %) 
    df = pd.read_csv(CPI_PATH, parse_dates=['date'], index_col='date')
    s  = df['cpi_yoy'].dropna()
    s.name = 'CPI Inflation'
    return s


def fx():
    #Monthly PKR/USD exchange rate 
    df = pd.read_csv(FX_PATH)
    df['date'] = pd.to_datetime(df['Month'], format='%b-%y')
    s = df.set_index('date')['PKR_USD'].dropna()
    s.name = 'FX Rate'
    return s


def cabalance():
    #Quarterly Current Account Balance (Million USD)
    df   = pd.read_csv(CA_PATH)
    row  = df[df['ITEMS'].str.strip().str.startswith('. Current account')]
    cols = [c for c in df.columns if c not in ('ITEMS', 'UNIT')]
    s    = row[cols].T.squeeze().astype(float)
    s.index = pd.to_datetime(s.index, format='%b-%Y')
    s    = s.sort_index()
    s.name = 'Current Account'
    return s


def gdp():
#Annual GDP growth rate %
    df   = pd.read_csv(GDP_PATH)
    row  = df[df['ITEMS'].str.contains('Growth Rate', na=False)]
    cols = [c for c in df.columns if c not in ('ITEMS', 'UNIT')]
    s    = row[cols].T.squeeze().astype(float)
    s.index = s.index.astype(int)
    s.name = 'GDP Growth'
    return s