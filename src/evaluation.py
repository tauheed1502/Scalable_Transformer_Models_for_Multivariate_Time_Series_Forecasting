import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from config import config

class ModelEvaluator:
    def evaluate_all_models(self, baseline_results, transformer_results, test_data):
        """Compare all models and generate performance metrics"""
        evaluation_results = {}
        
        # Calculate metrics for each series
        for series_id in test_data.keys():
            series_metrics = {}
            
            # Baseline metrics
            if series_id in baseline_results:
                for model_name, model_data in baseline_results[series_id].items():
                    metrics = self._calculate_metrics(
                        test_data[series_id]['Close'].values,
                        model_data['predictions']
                    )
                    series_metrics[model_name] = metrics
            
            # Transformer metrics
            if series_id in transformer_results:
                for model_name, model_data in transformer_results[series_id].items():
                    metrics = self._calculate_metrics(
                        test_data[series_id]['Close'].values,
                        model_data['predictions']
                    )
                    series_metrics[model_name] = metrics
            
            evaluation_results[series_id] = series_metrics
        
        # Aggregate results
        summary_results = self._aggregate_results(evaluation_results)
        
        # Save results
        self._save_results(evaluation_results, summary_results)
        
        return summary_results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
    
    def _aggregate_results(self, evaluation_results):
        """Aggregate results across all series"""
        all_metrics = {}
        
        for series_id, series_metrics in evaluation_results.items():
            for model_name, metrics in series_metrics.items():
                if model_name not in all_metrics:
                    all_metrics[model_name] = {metric: [] for metric in metrics.keys()}
                
                for metric, value in metrics.items():
                    all_metrics[model_name][metric].append(value)
        
        # Calculate averages
        summary = {}
        for model_name, metrics in all_metrics.items():
            summary[model_name] = {
                metric: np.mean(values) for metric, values in metrics.items()
            }
        
        return summary
    
    def generate_final_report(self, results):
        """Generate comprehensive performance report"""
        # Create performance comparison table
        df = pd.DataFrame(results).T
        df = df.round(3)
        
        # Highlight improvements
        if 'tft' in df.index and 'arima' in df.index:
            mape_improvement = (df.loc['arima', 'MAPE'] - df.loc['tft', 'MAPE']) / df.loc['arima', 'MAPE'] * 100
            rmse_improvement = (df.loc['arima', 'RMSE'] - df.loc['tft', 'RMSE']) / df.loc['arima', 'RMSE'] * 100
            
            print(f"TFT vs ARIMA:")
            print(f"MAPE Improvement: {mape_improvement:.1f}%")
            print(f"RMSE Improvement: {rmse_improvement:.1f}%")
        
        # Save to CSV
        df.to_csv(f"{config.RESULTS_PATH}model_performance.csv")
        
        return df
