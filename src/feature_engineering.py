"""
Advanced Feature Engineering Module
Implements volatility, regime shifts, Fourier encodings, and lagged correlations
"""

import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from arch import arch_model
import ta
from config import config

class FeatureEngineer:
    def engineer_features(self, data_dict):
        """Create advanced features for stable training under non-stationarity"""
        featured_data = {}
        
        for series_id, df in data_dict.items():
            # Volatility features
            df = self._add_volatility_features(df)
            
            # Regime shift detection
            df = self._detect_regime_shifts(df)
            
            # Fourier encodings for seasonality
            df = self._add_fourier_features(df)
            
            # Lagged correlations
            df = self._add_lagged_features(df)
            
            # Technical indicators for financial data
            if 'financial' in series_id:
                df = self._add_technical_indicators(df)
            
            featured_data[series_id] = df
        
        return featured_data
    
    def _add_volatility_features(self, df):
        """Add GARCH and rolling volatility measures"""
        if 'Returns' in df.columns:
            # Rolling volatility
            df['volatility_20'] = df['Returns'].rolling(config.VOLATILITY_WINDOW).std()
            df['volatility_60'] = df['Returns'].rolling(60).std()
            
            # GARCH volatility (simplified)
            try:
                returns = df['Returns'].dropna() * 100  # Scale for numerical stability
                if len(returns) > 100:
                    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
                    garch_fit = garch_model.fit(disp='off')
                    df['garch_volatility'] = garch_fit.conditional_volatility / 100
            except:
                df['garch_volatility'] = df['volatility_20']
        
        return df
    
    def _add_fourier_features(self, df):
        """Add Fourier encodings for seasonal patterns"""
        n = len(df)
        for i in range(1, config.FOURIER_TERMS + 1):
            df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * np.arange(n) / 252)  # 252 trading days
            df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * np.arange(n) / 252)
        
        return df
