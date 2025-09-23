# ===============================
# Cell 1: Imports and Setup
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.append('../')
from config import config
from src.data_preprocessing import DataCollector

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print(f'Exploring {len(config.FINANCIAL_TICKERS)} financial tickers and {len(config.MACRO_INDICATORS)} macro indicators')
print(f'Date range: {config.START_DATE} to {config.END_DATE}')


# ===============================
# Cell 2: Data Collection
# ===============================

# Initialize data collector
collector = DataCollector()

# Collect financial data
print('Collecting financial data...')
financial_data = collector.collect_financial_data()

print(f'Successfully collected {len(financial_data)} financial series')
print(f'Failed to collect: {len(config.FINANCIAL_TICKERS) - len(financial_data)} series')

# Display data summary
data_summary = []
for ticker, data in financial_data.items():
    data_summary.append({
        'Ticker': ticker,
        'Records': len(data),
        'Start_Date': data.index.min().strftime('%Y-%m-%d'),
        'End_Date': data.index.max().strftime('%Y-%m-%d'),
        'Missing_Values': data.isnull().sum().sum(),
        'Avg_Price': data['Close'].mean().round(2),
        'Volatility': (data['Close'].pct_change().std() * 252**0.5 * 100).round(2)
    })

summary_df = pd.DataFrame(data_summary)
print('\nData Summary Statistics:')
print(summary_df.head(10))


# ===============================
# Cell 3: Stationarity Analysis
# ===============================

from statsmodels.tsa.stattools import adfuller

# Test stationarity across financial series
stationarity_results = []

for ticker, data in list(financial_data.items())[:10]:  # Test first 10 series
    # Test price levels
    price_adf = adfuller(data['Close'].dropna())
    
    # Test returns
    returns_adf = adfuller(data['Returns'].dropna())
    
    stationarity_results.append({
        'Ticker': ticker,
        'Price_ADF_Stat': round(price_adf[0], 3),
        'Price_p_value': round(price_adf[1], 3),
        'Price_Stationary': 'Yes' if price_adf[1] < 0.05 else 'No',
        'Returns_ADF_Stat': round(returns_adf[0], 3),
        'Returns_p_value': round(returns_adf[1], 3),
        'Returns_Stationary': 'Yes' if returns_adf[1] < 0.05 else 'No'
    })

stationarity_df = pd.DataFrame(stationarity_results)
print('Stationarity Test Results (ADF Test):')
print(stationarity_df)


# ===============================
# Cell 4: Correlation Analysis
# ===============================

# Create correlation matrix for major stocks
major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']

# Collect returns for correlation analysis
returns_data = {}
for ticker in major_stocks:
    if ticker in financial_data:
        returns_data[ticker] = financial_data[ticker]['Returns']

returns_df = pd.DataFrame(returns_data)
correlation_matrix = returns_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Cross-Series Return Correlations\n(Key for Lagged Correlation Features)')
plt.tight_layout()
plt.savefig('../results/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

avg_corr = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean().round(3)
print('Average cross-correlation:', avg_corr)
