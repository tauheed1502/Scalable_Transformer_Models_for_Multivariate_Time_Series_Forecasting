"""
Model implementations for TFT, Informer, and baseline models
"""

import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from config import config

class ModelManager:
    def __init__(self):
        self.models = {}
    
    def create_tft_model(self, dataset):
        """Create Temporal Fusion Transformer model"""
        tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=config.LEARNING_RATE,
            hidden_size=config.TFT_PARAMS['hidden_size'],
            attention_head_size=config.TFT_PARAMS['attention_head_size'],
            dropout=config.TFT_PARAMS['dropout'],
            hidden_continuous_size=config.TFT_PARAMS['hidden_continuous_size'],
            loss=torch.nn.MSELoss(),
            reduce_on_plateau_patience=config.PATIENCE
        )
        return tft
    
    def create_informer_model(self, input_shape):
        """Create Informer model architecture"""
        # Simplified Informer implementation
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                config.INFORMER_PARAMS['d_model'], 
                return_sequences=True,
                dropout=config.INFORMER_PARAMS['dropout']
            ),
            tf.keras.layers.MultiHeadAttention(
                num_heads=config.INFORMER_PARAMS['n_heads'],
                key_dim=config.INFORMER_PARAMS['d_model']
            ),
            tf.keras.layers.Dense(config.FORECAST_HORIZON)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
            loss='mse',
            metrics=['mae', 'mape']
        )
        return model

class BaselineModels:
    @staticmethod
    def train_arima(data, order=(2,1,2)):
        """Train ARIMA model"""
        model = ARIMA(data, order=order)
        return model.fit()
    
    @staticmethod 
    def train_var(data, maxlags=5):
        """Train Vector Autoregression model"""
        model = VAR(data)
        return model.fit(maxlags=maxlags)
