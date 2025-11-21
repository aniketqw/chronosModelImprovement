# Chronos Model Improvement

This repository contains improvements to the Chronos time series forecasting model.

## DistanceAware/v1

Distance-aware enhancement for Chronos model.

**Related Repository:** [dm_eval_3 - Distance Aware v1](https://github.com/aniketqw/dm_eval_3/tree/master/improvement/p3_model/distanceAware/v1)

### Files
- `distance_aware_chronos.py` - Distance-aware Chronos implementation
- `compareModel.py` - Model comparison benchmark script
- `benchmark_results/` - Evaluation results

### Results Summary
| Metric | Distance-Aware | Original | Winner |
|--------|---------------|----------|--------|
| MAE | 1315.20 | 1309.50 | Original |
| RMSE | 1593.91 | 1584.71 | Original |

**Win Rate:** Distance-Aware wins 739/1484 series (49.8%)
