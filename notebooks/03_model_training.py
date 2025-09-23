# 03_model_training.py

# ===============================
# Cell 1: Setup and Imports
# ===============================

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.append('../')
from config import config
from src.models import ModelManager, BaselineModels
from src.training import TrainingPipeline

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training device: {device}')
print(f'PyTorch version: {torch.__version__}')


# ===============================
# Cell 2: Data Preparation
# ===============================

import glob
import os

featured_files = glob.glob('../data/processed/featured_*.csv')
print(f'Found {len(featured_files)} featured datasets')

featured_data = {}
for file_path in featured_files:
    series_id = os.path.basename(file_path).replace('featured_', '').replace('.csv', '')
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if len(df) > 200:  # Minimum length for training
            featured_data[series_id] = df
    except Exception as e:
        print(f'Failed to load {series_id}: {e}')

print(f'Loaded {len(featured_data)} series for training')


def prepare_training_data(data_dict, sequence_length=60, forecast_horizon=20):
    """Prepare data for transformer training"""
    training_data = []
    
    for series_id, df in data_dict.items():
        df_clean = df.fillna(method='ffill').fillna(method='bfill')
        
        # Add time features
        df_clean['time_idx'] = range(len(df_clean))
        df_clean['series_id'] = series_id
        
        # Target variable
        if 'Returns' in df_clean.columns:
            df_clean['target'] = df_clean['Returns']
        elif 'Close' in df_clean.columns:
            df_clean['target'] = df_clean['Close'].pct_change()
        else:
            continue
        
        df_clean = df_clean.dropna(subset=['target'])
        
        if len(df_clean) >= sequence_length + forecast_horizon:
            training_data.append(df_clean)
    
    return pd.concat(training_data, ignore_index=True)


# Prepare data
training_data = prepare_training_data(
    featured_data, config.SEQUENCE_LENGTH, config.FORECAST_HORIZON
)
print(f'Combined training data shape: {training_data.shape}')


# ===============================
# Cell 3: TFT Model Training
# ===============================

def create_tft_dataset(data, max_encoder_length=60, max_prediction_length=20):
    """Create TimeSeriesDataSet for TFT training"""
    time_varying_known_reals = ['time_idx']
    time_varying_unknown_reals = ['target']
    
    # Dynamically select numeric features
    available_features = [
        col for col in data.columns
        if col not in ['series_id', 'time_idx', 'target']
        and data[col].dtype in [np.float64, np.float32]
    ]
    
    available_features = available_features[:30]  # limit features
    time_varying_unknown_reals.extend(available_features)
    
    dataset = TimeSeriesDataSet(
        data,
        time_idx='time_idx',
        target='target',
        group_ids=['series_id'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=['series_id']),
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    return dataset


# Train/validation split
train_size = int(len(training_data) * 0.8)
train_data = training_data.iloc[:train_size]
val_data = training_data.iloc[train_size:]

train_dataset = create_tft_dataset(train_data)
val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_data)

train_dataloader = train_dataset.to_dataloader(
    train=True, batch_size=config.BATCH_SIZE, num_workers=0
)
val_dataloader = val_dataset.to_dataloader(
    train=False, batch_size=config.BATCH_SIZE, num_workers=0
)

print(f'TFT datasets created: Train={len(train_dataset)}, Val={len(val_dataset)}')


# ===============================
# Cell 4: Train TFT Model
# ===============================

tft_model = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=config.LEARNING_RATE,
    hidden_size=config.TFT_PARAMS['hidden_size'],
    attention_head_size=config.TFT_PARAMS['attention_head_size'],
    dropout=config.TFT_PARAMS['dropout'],
    hidden_continuous_size=config.TFT_PARAMS['hidden_continuous_size'],
    output_size=7,  # 7 quantiles
    loss=QuantileLoss(),
    reduce_on_plateau_patience=config.PATIENCE,
)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=config.PATIENCE,
    verbose=False,
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=50,  # Reduced for demo
    accelerator='auto',
    callbacks=[early_stop_callback],
    enable_progress_bar=True,
)

trainer.fit(tft_model, train_dataloader, val_dataloader)

print(f'TFT training completed in {trainer.current_epoch} epochs')
print(f'Best validation loss: {trainer.callback_metrics['val_loss'].item():.4f}')

torch.save(tft_model.state_dict(), '../saved_models/tft_model.pth')
print('TFT model saved')


# ===============================
# Cell 5: Informer Model
# ===============================

def create_informer_model(input_shape, forecast_horizon, d_model=512, n_heads=8):
    """Create simplified Informer model using TensorFlow"""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(d_model)(inputs)
    
    for _ in range(2):  # two encoder layers
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=0.1
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_output)
        
        ff_output = tf.keras.layers.Dense(d_model * 4, activation='relu')(x)
        ff_output = tf.keras.layers.Dense(d_model)(ff_output)
        ff_output = tf.keras.layers.Dropout(0.1)(ff_output)
        x = tf.keras.layers.LayerNormalization()(x + ff_output)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(forecast_horizon)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def prepare_informer_data(data, sequence_length, forecast_horizon):
    """Prepare data for Informer model training"""
    X, y = [], []
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['time_idx']]
    numeric_cols = numeric_cols[:20]
    
    for series_id in data['series_id'].unique():
        series_data = data[data['series_id'] == series_id].copy()
        series_data = series_data.sort_values('time_idx')
        
        features = series_data[numeric_cols].fillna(0).values
        target = series_data['target'].fillna(0).values
        
        for i in range(len(features) - sequence_length - forecast_horizon + 1):
            X.append(features[i:i + sequence_length])
            y.append(target[i + sequence_length:i + sequence_length + forecast_horizon])
    
    return np.array(X), np.array(y)


# Prepare Informer data
X_train, y_train = prepare_informer_data(train_data, config.SEQUENCE_LENGTH, config.FORECAST_HORIZON)
X_val, y_val = prepare_informer_data(val_data, config.SEQUENCE_LENGTH, config.FORECAST_HORIZON)

print(f'Informer data prepared: Train X{X_train.shape}, y{y_train.shape}')

informer_model = create_informer_model(
    input_shape=(config.SEQUENCE_LENGTH, X_train.shape[2]),
    forecast_horizon=config.FORECAST_HORIZON,
)

informer_model.compile(
    optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
    loss='mse',
    metrics=['mae', 'mape']
)

history = informer_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,  # Reduced for demo
    batch_size=config.BATCH_SIZE,
    verbose=1
)

informer_model.save('../saved_models/informer_model.h5')
print('Informer model saved')
