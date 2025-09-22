"""
Configuration file for Large-Scale Time Series Forecasting project
Contains all hyperparameters, data paths, and model configurations
"""

import os
from datetime import datetime, timedelta

class Config:
    # Project metadata
    PROJECT_NAME = "Large-Scale Time Series Forecasting with Transformers"
    VERSION = "1.0.0"
    AUTHOR = "Your Name"
    
    # Random seed for reproducibility
    SEED = 42
    
    # Data configuration
    START_DATE = "2020-01-01"
    END_DATE = "2025-06-30"
    
    # Financial tickers (50+ series)
    FINANCIAL_TICKERS = [
        # Tech stocks
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM',
        # Banking & Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'MRK', 'LLY',
        # Consumer & Retail
        'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
        # Energy & Industrials
        'XOM', 'CVX', 'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT',
        # Additional diversification
        'ORCL', 'INTC', 'AMD', 'IBM', 'QCOM'
    ]
    
    # Macroeconomic indicators from FRED
    MACRO_INDICATORS = {
        'GDP': 'GDP',
        'UNEMPLOYMENT': 'UNRATE',
        'CPI': 'CPIAUCSL',
        'FEDERAL_FUNDS_RATE': 'FEDFUNDS',
        'VIX': 'VIXCLS',
        'DGS10': 'DGS10',  # 10-Year Treasury
        'DGS3MO': 'DGS3MO',  # 3-Month Treasury
        'M2_MONEY_SUPPLY': 'M2SL',
        'INDUSTRIAL_PRODUCTION': 'INDPRO',
        'RETAIL_SALES': 'RSAFS'
    }
    
    # API Keys (set as environment variables)
    FRED_API_KEY = os.getenv('FRED_API_KEY', 'your_fred_api_key_here')
    
    # Data paths
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
    PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
    RESULTS_PATH = os.path.join(BASE_PATH, 'results')
    PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')
    PREDICTIONS_PATH = os.path.join(RESULTS_PATH, 'predictions')
    MODELS_PATH = os.path.join(BASE_PATH, 'saved_models')
    
    # Feature engineering parameters
    VOLATILITY_WINDOW = 20
    FOURIER_TERMS = 10
    LAG_FEATURES = [1, 2, 3, 5, 10, 20]
    
    # Model architecture parameters
    SEQUENCE_LENGTH = 60
    FORECAST_HORIZON = 20
    
    # TFT Parameters
    TFT_PARAMS = {
        'hidden_size': 128,
        'attention_head_size': 4,
        'dropout': 0.2,
        'hidden_continuous_size': 16,
        'output_size': 7,  # Quantiles
        'loss': 'QuantileLoss',
        'reduce_on_plateau_patience': 4
    }
    
    # Informer Parameters  
    INFORMER_PARAMS = {
        'd_model': 512,
        'd_ff': 2048,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'factor': 5,
        'dropout': 0.1,
        'activation': 'gelu'
    }
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 100
    PATIENCE = 10
    
    # Data splits
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.2
    TEST_SIZE = 0.1
    
    # Performance thresholds (for validation)
    TARGET_RMSE_IMPROVEMENT = 20.0  # 20% improvement target
    TARGET_MAPE_IMPROVEMENT = 27.0  # 27% improvement target
    
    # Visualization settings
    FIGURE_SIZE = (15, 10)
    DPI = 300
    STYLE = 'seaborn-v0_8'
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create global config instance
config = Config()

# Create directories if they don't exist
for path in [config.DATA_PATH, config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH, 
             config.RESULTS_PATH, config.PLOTS_PATH, config.PREDICTIONS_PATH, config.MODELS_PATH]:
    os.makedirs(path, exist_ok=True)

# Export key configurations
__all__ = ['config']
