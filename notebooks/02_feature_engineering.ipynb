# 02_feature_engineering.py

# ===============================
# Cell 1: Setup and Data Loading
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from hmmlearn import hmm
import ta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.append('../')
from config import config
from src.data_preprocessing import DataCollector, DataPreprocessor
from src.feature_engineering import FeatureEngineer

print('🔧 Advanced Feature Engineering Pipeline')
print(f'Target: {len(config.FINANCIAL_TICKERS) + len(config.MACRO_INDICATORS)} time series')

# Load preprocessed data
collector = DataCollector()
raw_data = collector.collect_all_data()

preprocessor = DataPreprocessor()
clean_data = preprocessor.preprocess_data(raw_data)

print(f'Loaded {len(clean_data)} preprocessed time series')


# ===============================
# Cell 2: Volatility Features
# ===============================

def add_volatility_features(df, series_id):
    """Add comprehensive volatility measures"""
    if 'Returns' not in df.columns:
        return df
    
    # Rolling volatility (multiple windows)
    df['volatility_5'] = df['Returns'].rolling(5).std()
    df['volatility_20'] = df['Returns'].rolling(config.VOLATILITY_WINDOW).std()
    df['volatility_60'] = df['Returns'].rolling(60).std()
    
    # Volatility of volatility
    df['vol_of_vol'] = df['volatility_20'].rolling(20).std()
    
    # GARCH(1,1) volatility
    try:
        returns = df['Returns'].dropna() * 100  # Scale for numerical stability
        if len(returns) > 100:
            garch_model = arch_model(returns, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            garch_vol = garch_fit.conditional_volatility / 100
            
            # Align with original dataframe
            garch_series = pd.Series(index=df.index, dtype=float)
            garch_series.loc[garch_vol.index] = garch_vol.values
            df['garch_volatility'] = garch_series
        else:
            df['garch_volatility'] = df['volatility_20']
    except Exception as e:
        print(f'GARCH failed for {series_id}: {e}')
        df['garch_volatility'] = df['volatility_20']
    
    return df

# Apply to sample series
sample_key = list(clean_data.keys())[0]
sample_data = clean_data[sample_key].copy()
featured_sample = add_volatility_features(sample_data, sample_key)

print('Volatility features added:')
vol_features = [col for col in featured_sample.columns if 'vol' in col.lower()]
print(vol_features)


# ===============================
# Cell 3: Regime Detection
# ===============================

def detect_regime_shifts(df, n_regimes=3):
    """Detect regime shifts using Hidden Markov Model"""
    if 'Returns' not in df.columns:
        return df
    
    returns = df['Returns'].dropna().values.reshape(-1, 1)
    
    if len(returns) < 100:
        df['regime'] = 1
        df['regime_prob'] = 0.5
        return df
    
    try:
        # Fit HMM
        model = hmm.GaussianHMM(n_components=n_regimes, covariance_type='full', random_state=42)
        model.fit(returns)
        
        # Predict regimes
        hidden_states = model.predict(returns)
        state_probs = model.predict_proba(returns)
        
        # Align with original dataframe
        regime_series = pd.Series(index=df.index, dtype=int)
        prob_series = pd.Series(index=df.index, dtype=float)
        
        valid_returns_idx = df['Returns'].dropna().index
        regime_series.loc[valid_returns_idx] = hidden_states
        prob_series.loc[valid_returns_idx] = state_probs.max(axis=1)
        
        # Forward fill for missing values
        df['regime'] = regime_series.fillna(method='ffill').fillna(1)
        df['regime_prob'] = prob_series.fillna(method='ffill').fillna(0.5)
        
        # Regime transition indicator
        df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
        
    except Exception as e:
        print(f'HMM failed: {e}')
        df['regime'] = 1
        df['regime_prob'] = 0.5
        df['regime_change'] = 0
    
    return df

# Apply regime detection
regime_sample = detect_regime_shifts(featured_sample)
print(f'Regime distribution: {regime_sample["regime"].value_counts().to_dict()}')


# ===============================
# Cell 4: Fourier Features
# ===============================

def add_fourier_features(df, n_terms=10):
    """Add Fourier encoding features for seasonal patterns"""
    n = len(df)
    
    for i in range(1, n_terms + 1):
        # Annual cycle (252 trading days)
        df[f'fourier_sin_annual_{i}'] = np.sin(2 * np.pi * i * np.arange(n) / 252)
        df[f'fourier_cos_annual_{i}'] = np.cos(2 * np.pi * i * np.arange(n) / 252)
        
        # Monthly cycle (21 trading days)
        df[f'fourier_sin_monthly_{i}'] = np.sin(2 * np.pi * i * np.arange(n) / 21)
        df[f'fourier_cos_monthly_{i}'] = np.cos(2 * np.pi * i * np.arange(n) / 21)
    
    return df

# Apply Fourier features
fourier_sample = add_fourier_features(regime_sample, n_terms=config.FOURIER_TERMS)

fourier_features = [col for col in fourier_sample.columns if 'fourier' in col]
print(f'Added {len(fourier_features)} Fourier features')
print('Sample Fourier features:', fourier_features[:6])
