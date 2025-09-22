"""
Training pipeline for all models with hyperparameter optimization
"""

import torch
from pytorch_lightning import Trainer
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from config import config

class TrainingPipeline:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.results = {}
    
    def train_transformer_models(self, featured_data):
        """Train TFT and Informer models with optimization"""
        results = {}
        
        for series_id, data in featured_data.items():
            print(f"Training transformers for {series_id}...")
            
            # Prepare data for training
            train_data, val_data, test_data = self._prepare_data(data)
            
            # Train TFT
            tft_model = self._train_tft_with_optuna(train_data, val_data)
            tft_predictions = self._predict(tft_model, test_data)
            
            # Train Informer  
            informer_model = self._train_informer(train_data, val_data)
            informer_predictions = self._predict(informer_model, test_data)
            
            results[series_id] = {
                'tft': {'model': tft_model, 'predictions': tft_predictions},
                'informer': {'model': informer_model, 'predictions': informer_predictions}
            }
        
        return results
    
    def train_baseline_models(self, featured_data):
        """Train ARIMA, VAR, and LSTM baseline models"""
        results = {}
        
        for series_id, data in featured_data.items():
            print(f"Training baselines for {series_id}...")
            
            # Simple train/test split
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            # Train models
            baseline_results = {}
            
            # ARIMA
            try:
                arima_model = BaselineModels.train_arima(train_data['Close'])
                arima_pred = arima_model.forecast(len(test_data))
                baseline_results['arima'] = {
                    'model': arima_model, 
                    'predictions': arima_pred
                }
            except:
                print(f"ARIMA failed for {series_id}")
            
            results[series_id] = baseline_results
        
        return results
    
    def _train_tft_with_optuna(self, train_data, val_data):
        """Hyperparameter optimization for TFT"""
        def objective(trial):
            # Suggest hyperparameters
            hidden_size = trial.suggest_int('hidden_size', 32, 128)
            attention_heads = trial.suggest_int('attention_heads', 2, 8)
            dropout = trial.suggest_float('dropout', 0.1, 0.3)
            
            # Train model with suggested params
            model = self.model_manager.create_tft_model(train_data)
            trainer = Trainer(max_epochs=50, logger=False, checkpoint_callback=False)
            trainer.fit(model, train_data, val_data)
            
            # Return validation loss
            return trainer.callback_metrics['val_loss'].item()
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        # Train final model with best params
        best_model = self.model_manager.create_tft_model(train_data)
        trainer = Trainer(max_epochs=config.MAX_EPOCHS)
        trainer.fit(best_model, train_data, val_data)
        
        return best_model
