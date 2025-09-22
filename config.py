"""
Explainability analysis with attention maps and SHAP
"""

import shap
import matplotlib.pyplot as plt
import torch
import numpy as np
from config import config

class ExplainabilityAnalyzer:
    def generate_attention_maps(self, transformer_results):
        """Generate attention visualization for transformer models"""
        for series_id, models in transformer_results.items():
            if 'tft' in models:
                model = models['tft']['model']
                
                # Extract attention weights
                with torch.no_grad():
                    attention_weights = model.attention_weights
                
                # Create attention heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(attention_weights.cpu().numpy(), cmap='Blues')
                ax.set_title(f'Attention Map - {series_id}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Features')
                plt.colorbar(im)
                
                plt.savefig(f"{config.PLOTS_PATH}attention_{series_id}.png")
                plt.close()
    
    def perform_shap_analysis(self, transformer_results, featured_data):
        """SHAP analysis for model interpretability"""
        for series_id, models in transformer_results.items():
            if 'tft' in models and series_id in featured_data:
                model = models['tft']['model']
                data = featured_data[series_id]
                
                # Create SHAP explainer
                explainer = shap.DeepExplainer(model, data.values[:100])
                shap_values = explainer.shap_values(data.values[100:200])
                
                # Generate SHAP plots
                shap.summary_plot(
                    shap_values[0], 
                    data.columns,
                    show=False,
                    max_display=10
                )
                plt.savefig(f"{config.PLOTS_PATH}shap_{series_id}.png")
                plt.close()
