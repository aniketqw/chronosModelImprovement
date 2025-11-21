# Distance-Aware Chronos: v1 vs v2 Comparison

## Key Differences

| Aspect | v1 | v2 (soft_label) |
|--------|-----|-----------------|
| **Device** | CPU only | MPS (M3 GPU) auto-detect |
| **Loss Function** | Hard labels (one-hot) | Soft labels (Gaussian smoothing) |
| **Speed** | ~1.2 it/s | ~1.7-2.0 it/s (~40-60% faster) |
| **File** | `distance_aware_chronos.py` | `distance_aware_chronos.py` + `distance_aware_loss()` |

---

## v2 Benchmark Results

From `v2_soft_label/benchmark_results/comparison.csv`:

| Metric | Distance-Aware | Original | Improvement | Winner |
|--------|----------------|----------|-------------|--------|
| **MAE** | 1305.0451 | 1309.2933 | **+0.32%** | 🏆 DA |
| **RMSE** | 1582.1674 | 1583.0782 | **+0.06%** | 🏆 DA |
| **MAPE** | 26775627.34 | 26649857.74 | -0.47% | Original |

**Distance-Aware wins 2/3 metrics (MAE and RMSE)**

---

## How Both Use `Phoenix21/distance-aware-chronos-t`

### Step 1: Download from HuggingFace
```python
# Both v1 and v2 do this identically
from huggingface_hub import hf_hub_download

config_path = hf_hub_download(
    repo_id="Phoenix21/distance-aware-chronos-t",
    filename="config.json"
)
distance_output_path = hf_hub_download(
    repo_id="Phoenix21/distance-aware-chronos-t",
    filename="distance_output.pt"
)
```

### Step 2: Load Base Chronos Model
```python
# Both load amazon/chronos-t5-small as base
self.base_model = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map=device,
    dtype=torch.float32
)
```

### Step 3: Load Trained Distance Output Weights
```python
# Both load the same trained weights
state_dict = torch.load(distance_output_path, map_location=device)
self.da_model.distance_output.load_state_dict(state_dict)
```

---

## What's New in v2

### 1. MPS Auto-Detection (Apple Silicon)
```python
def get_device():
    if torch.backends.mps.is_available():
        return 'mps'  # Uses M3 GPU
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
```

### 2. Soft Label Loss Function (Gaussian Smoothing)
```python
def distance_aware_loss(logits, target_bin, vocab_size=4096, sigma=2.0):
    """
    Instead of one-hot encoding the target bin, create a soft distribution
    where bins closer to the target have higher probability mass.
    """
    device = logits.device
    bins = torch.arange(vocab_size, device=device, dtype=torch.float32)

    # Gaussian soft labels centered at target bin
    soft_labels = torch.exp(-((bins - target_bin)**2) / (2 * sigma**2))
    soft_labels = soft_labels / soft_labels.sum()

    # Cross entropy with soft labels
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(soft_labels * log_probs)
```

**Why Soft Labels?**
- Hard labels: `[0, 0, 0, 1, 0, 0, 0]` - only exact bin is correct
- Soft labels: `[0.01, 0.05, 0.24, 0.40, 0.24, 0.05, 0.01]` - nearby bins get partial credit
- Result: Smoother gradients, better handling of ordinal nature of bins

---

## Phoenix21/distance-aware-chronos-t: Trained Model Details

### Model Architecture

**Base Model**: `amazon/chronos-t5-small` (T5ForConditionalGeneration)
- All T5 parameters are **FROZEN** during training
- Only the DistanceAwareOutputLayer is trained

**DistanceAwareOutputLayer Components**:
```
Input: hidden_states (d_model from T5)
    │
    ├── value_projection: Linear(hidden_size → 1)
    │       → Predicts continuous value
    │
    ├── confidence_projection: Linear(hidden_size → 1) + Sigmoid
    │       → Confidence score for mixing
    │
    ├── gaussian_centers: Parameter [1, 4096] initialized linspace(-15, 15)
    ├── gaussian_widths: Parameter [4096] initialized to 0.5
    │       → Soft binning via Gaussian kernels
    │
    ├── ordinal_embed: Embedding(4096, 64)
    │       → Sinusoidal position encoding initialization
    │
    └── mix_layer: Linear(hidden_size+65 → hidden_size) → ReLU → Dropout(0.1) → Linear(→ 4096)
            → Final logits

Output: (1-confidence)*gaussian_logits + confidence*mix_logits
```

### Loss Function (Combined)

The model was trained with a **weighted combination of 3 losses**:

```python
loss = 0.5 * ordinal_cross_entropy + 0.3 * smooth_label_loss + 0.2 * earth_movers_distance
```

| Loss | Weight | Description |
|------|--------|-------------|
| **Ordinal Cross-Entropy** | 0.5 | CE + `log(1 + distance)` penalty |
| **Smooth Label Loss** | 0.3 | KL divergence with Gaussian soft labels |
| **Earth Mover's Distance** | 0.2 | L1 distance between CDFs (Wasserstein) |

### Tokenization

```python
# Mean scaling
scale = np.abs(time_series).mean() + 1e-10
scaled = time_series / scale

# Clip to range
scaled = np.clip(scaled, -15, 15)

# Quantize to 4096 bins
bins = np.linspace(-15, 15, 4096)
tokens = np.digitize(scaled, bins) - 1
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Scheduler** | CosineAnnealingLR |
| **Epochs** | 10 |
| **Batch Size** | 8 |
| **Gradient Clipping** | 1.0 |
| **Context/Label Split** | 80% / 20% |
| **Max Sequence Length** | 512 |
| **Num Bins** | 4096 |

### Trainable Parameters

Only the following are trained (T5 is frozen):
- `distance_output.gaussian_centers`
- `distance_output.gaussian_widths`
- `distance_output.ordinal_embed`
- `distance_output.value_projection`
- `distance_output.confidence_projection`
- `distance_output.mix_layer`
- `temperature` (learnable scalar)

### OrdinalLoss Details

**1. Ordinal Cross-Entropy**:
```python
ce_loss = F.cross_entropy(logits, targets)
distances = distance_matrix[predictions, targets]  # Normalized 0-1
distance_penalty = log(1 + distances) * alpha
total = ce_loss + distance_penalty
```

**2. Smooth Label Loss**:
```python
# Gaussian-like distribution centered at target
distances = |bin_indices - target|
smooth_labels = exp(-distances * beta)
smooth_labels = smooth_labels / sum(smooth_labels)
loss = KL_divergence(log_softmax(logits), smooth_labels)
```

**3. Earth Mover's Distance**:
```python
cdf_pred = cumsum(softmax(logits))
cdf_target = cumsum(one_hot(target))
emd = sum(|cdf_pred - cdf_target|)
```

---

## Architecture (Same in Both)

```
Phoenix21/distance-aware-chronos-t
├── config.json          → Training metadata (epoch 10, val_loss: 2.4703)
└── distance_output.pt   → Trained DistanceOutput layer weights
        │
        ▼
┌─────────────────────────────────┐
│ DistanceOutput Layer            │
│ - mix_layer: 577 → 512 → 4096   │
│ - gaussian_centers, widths      │
│ - ordinal_embed                 │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ Base: amazon/chronos-t5-small   │
│ (Pretrained T5 for forecasting) │
└─────────────────────────────────┘
```

---

## Summary

| Feature | v1 | v2 |
|---------|-----|-----|
| Same trained weights | Yes | Yes |
| Same base model | Yes | Yes |
| M3 GPU acceleration | No | Yes |
| Soft label loss | No | Yes |
| Benchmark complete | No | Yes |

**Recommendation**: Use v2 for faster inference on Apple Silicon and the new soft-label loss for future fine-tuning.
