
"""
Streamlined External Benchmark: Distance-Aware Chronos vs Original
Compares models on external datasets with essential metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
from datasets import load_dataset
from chronos import ChronosPipeline
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation"""
    distance_aware_repo: str = "Phoenix21/distance-aware-chronos-t"  # HuggingFace repo
    original_repo: str = "amazon/chronos-t5-small"
    output_dir: str = "./benchmark_results"
    max_series: int = 500
    num_samples: int = 100
    device: str = None


class ModelLoader:
    """Handles loading and managing models"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.da_model = None
        self.orig_model = None
    
    def load_models(self) -> Tuple[object, object]:
        """Load both models from HuggingFace or local checkpoint"""
        print(f"\n{'='*60}\nLoading Models (Device: {self.device})\n{'='*60}")
        
        # Load Distance-Aware model
        print(f"1. Distance-Aware Chronos from {self.config.distance_aware_repo}...")
        from distance_aware_chronos import DistanceAwareChronos
        from huggingface_hub import hf_hub_download
        import json
        
        # Download config and weights from HuggingFace
        config_path = hf_hub_download(
            repo_id=self.config.distance_aware_repo,
            filename="config.json",
            repo_type="model"
        )
        distance_output_path = hf_hub_download(
            repo_id=self.config.distance_aware_repo,
            filename="distance_output.pt",
            repo_type="model"
        )
        
        # Load config
        with open(config_path, 'r') as f:
            da_config = json.load(f)
        
        # Initialize model
        self.da_model = DistanceAwareChronos(
            model_name=da_config.get('base_model', 'amazon/chronos-t5-small'),
            num_bins=da_config.get('num_bins', 4096),
            device=self.device
        )
        
        # Load trained distance output weights
        state_dict = torch.load(distance_output_path, map_location=self.device)
        self.da_model.distance_output.load_state_dict(state_dict)
        
        print(f"  ✓ Loaded (epoch {da_config.get('training_epoch', 'N/A')}, val_loss: {da_config.get('val_loss', 0):.4f})")
        
        # Load Original Chronos
        print("2. Original Chronos...")
        self.orig_model = ChronosPipeline.from_pretrained(
            self.config.original_repo,
            device_map=self.device,
            dtype=torch.float32
        )
        
        print("✓ Models loaded successfully\n")
        return self.da_model, self.orig_model


class BenchmarkEvaluator:
    """Core evaluation logic"""
    
    # External datasets configuration
    DATASETS = {
        'm4_quarterly': {'freq': 'Q', 'horizon': 8},
        'm4_yearly': {'freq': 'Y', 'horizon': 6},
        'monash_tourism_monthly': {'freq': 'M', 'horizon': 24},
        'monash_tourism_quarterly': {'freq': 'Q', 'horizon': 8},
        'monash_cif_2016': {'freq': 'M', 'horizon': 12},
        'monash_hospital': {'freq': 'M', 'horizon': 12},
        'monash_fred_md': {'freq': 'M', 'horizon': 12},
    }
    
    def __init__(self, da_model, orig_model, config: BenchmarkConfig):
        self.da_model = da_model
        self.orig_model = orig_model
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_datasets(self) -> List[Dict]:
        """Load external test datasets"""
        print(f"{'='*60}\nLoading External Datasets\n{'='*60}")
        
        all_data = []
        for dataset_name, params in self.DATASETS.items():
            print(f"Loading: {dataset_name}...")
            try:
                dataset = load_dataset(
                    "autogluon/chronos_datasets",
                    dataset_name,
                    trust_remote_code=True,
                    split='train'
                )
                
                count = 0
                for item in dataset:
                    if count >= self.config.max_series:
                        break
                    
                    series = np.array(item.get('target', item.get('value', [])), dtype=np.float32)
                    series = series[~np.isnan(series) & ~np.isinf(series)]
                    
                    if 100 <= len(series) <= 5000:
                        all_data.append({
                            'series': series,
                            'dataset': dataset_name,
                            'horizon': params['horizon']
                        })
                        count += 1
                
                print(f"  ✓ Loaded {count} series")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        print(f"\nTotal test series: {len(all_data)}\n")
        return all_data
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """Evaluate both models on test data"""
        print(f"{'='*60}\nEvaluating Models\n{'='*60}")
        
        results = {'da': [], 'orig': [], 'meta': []}
        
        for item in tqdm(test_data, desc="Forecasting"):
            series = item['series']
            horizon = item['horizon']
            
            # Prepare split
            split_idx = len(series) - horizon
            context = series[max(0, split_idx - 512):split_idx]
            truth = series[split_idx:split_idx + horizon]
            
            if len(context) < 50:
                continue
            
            # Metadata
            results['meta'].append({
                'dataset': item['dataset'],
                'horizon': horizon,
                'context_len': len(context)
            })
            
            # Evaluate both models
            da_pred = self._forecast(self.da_model, context, horizon, is_da=True)
            orig_pred = self._forecast(self.orig_model, context, horizon, is_da=False)
            
            results['da'].append(self._compute_metrics(da_pred, truth))
            results['orig'].append(self._compute_metrics(orig_pred, truth))
        
        return results
    
    def _forecast(self, model, context: np.ndarray, horizon: int, is_da: bool) -> np.ndarray:
        """Generate forecast"""
        try:
            with torch.no_grad():
                if is_da:
                    forecast = model.predict(context, horizon, self.config.num_samples)
                    return forecast
                else:
                    # Original Chronos expects predictions through pipeline
                    forecast = model.predict(
                        torch.tensor(context[np.newaxis, :], dtype=torch.float32),
                        prediction_length=horizon,
                        num_samples=self.config.num_samples
                    )
                    # Take median across samples
                    return np.median(forecast.cpu().numpy()[0], axis=0)
        except Exception as e:
            # Print first 5 errors for debugging
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            if self._error_count < 5:
                print(f"Forecast error ({type(e).__name__}): {e}")
                self._error_count += 1
            return np.full(horizon, np.nan)
    
    def _compute_metrics(self, forecast: np.ndarray, truth: np.ndarray) -> Dict:
        """Compute MAE, RMSE, MAPE"""
        if np.any(np.isnan(forecast)):
            return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
        
        min_len = min(len(forecast), len(truth))
        forecast, truth = forecast[:min_len], truth[:min_len]
        
        mae = np.mean(np.abs(forecast - truth))
        rmse = np.sqrt(np.mean((forecast - truth) ** 2))
        mape = np.mean(np.abs((truth - forecast) / (np.abs(truth) + 1e-10))) * 100
        
        return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}


class ResultsAnalyzer:
    """Generate reports and visualizations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        sns.set_style("whitegrid")
    
    def generate_report(self, results: Dict) -> pd.DataFrame:
        """Generate comparison report"""
        print(f"\n{'='*60}\nGenerating Report\n{'='*60}")
        
        # Filter valid results and ensure proper DataFrame creation
        da_valid = [r for r in results['da'] if isinstance(r, dict) and not np.isnan(r.get('MAE', np.nan))]
        orig_valid = [r for r in results['orig'] if isinstance(r, dict) and not np.isnan(r.get('MAE', np.nan))]
        
        if not da_valid or not orig_valid:
            print("⚠️ No valid results to analyze")
            return pd.DataFrame()
        
        da_df = pd.DataFrame(da_valid)
        orig_df = pd.DataFrame(orig_valid)
        
        print(f"Valid results: Distance-Aware={len(da_df)}, Original={len(orig_df)}")
        
        comparison = []
        for metric in ['MAE', 'RMSE', 'MAPE']:
            da_mean = da_df[metric].mean()
            orig_mean = orig_df[metric].mean()
            improvement = ((orig_mean - da_mean) / orig_mean) * 100
            
            comparison.append({
                'Metric': metric,
                'Distance-Aware': f"{da_mean:.4f}",
                'Original': f"{orig_mean:.4f}",
                'Improvement_%': f"{improvement:+.2f}",
                'Winner': '🏆 DA' if improvement > 0 else 'Original'
            })
        
        df = pd.DataFrame(comparison)
        df.to_csv(self.output_dir / "comparison.csv", index=False)
        
        print("\n" + df.to_string(index=False))
        
        # Win summary
        da_wins = sum(1 for d, o in zip(results['da'], results['orig']) 
                      if d.get('MAE', float('inf')) < o.get('MAE', float('inf')))
        total = len(results['da'])
        print(f"\n{'='*60}")
        print(f"🏆 Distance-Aware wins: {da_wins}/{total} ({da_wins/total*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return df
    
    def create_visualizations(self, results: Dict):
        """Create essential plots"""
        print("Creating visualizations...")
        
        da_valid = [r for r in results['da'] if isinstance(r, dict) and not np.isnan(r.get('MAE', np.nan))]
        orig_valid = [r for r in results['orig'] if isinstance(r, dict) and not np.isnan(r.get('MAE', np.nan))]
        
        if not da_valid or not orig_valid:
            print("⚠️ No valid results for visualization")
            return
        
        da_df = pd.DataFrame(da_valid)
        orig_df = pd.DataFrame(orig_valid)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot comparison
        for idx, metric in enumerate(['MAE', 'RMSE']):
            ax = axes[idx]
            data = [da_df[metric], orig_df[metric]]
            bp = ax.boxplot(data, labels=['Distance-Aware', 'Original'],
                           patch_artist=True, showmeans=True)
            
            for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {self.output_dir / 'comparison.png'}")


def main():
    """Main execution"""
    # Configuration
    config = BenchmarkConfig(
        distance_aware_repo="Phoenix21/distance-aware-chronos-t",  # HuggingFace model
        original_repo="amazon/chronos-t5-small",
        max_series=500,
        num_samples=100
    )
    
    # Load models
    loader = ModelLoader(config)
    da_model, orig_model = loader.load_models()
    
    # Evaluate
    evaluator = BenchmarkEvaluator(da_model, orig_model, config)
    test_data = evaluator.load_datasets()
    
    if not test_data:
        print("No test data loaded!")
        return
    
    results = evaluator.evaluate(test_data)
    
    # Analyze
    analyzer = ResultsAnalyzer(Path(config.output_dir))
    analyzer.generate_report(results)
    analyzer.create_visualizations(results)
    
    print(f"\n{'='*60}")
    print(f"✓ Benchmark Complete! Results in: {config.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()