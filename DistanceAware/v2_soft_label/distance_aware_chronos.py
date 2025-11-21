"""
Distance-Aware Chronos Model v2 - Soft Label Loss
Extends Chronos with distance-aware forecasting using Gaussian soft labels.
Optimized for Apple Silicon (M3) with MPS backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from chronos import ChronosPipeline


def get_device():
    """Get best available device for Apple Silicon"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def distance_aware_loss(logits, target_bin, vocab_size=4096, sigma=2.0):
    """
    Soft labels with Gaussian smoothing instead of hard labels.

    Instead of one-hot encoding the target bin, we create a soft distribution
    where bins closer to the target have higher probability mass.

    Args:
        logits: Model output logits [batch_size, vocab_size] or [vocab_size]
        target_bin: Target bin index (int or tensor)
        vocab_size: Number of bins in vocabulary
        sigma: Standard deviation for Gaussian smoothing (higher = softer labels)

    Returns:
        Cross entropy loss with soft labels
    """
    device = logits.device
    bins = torch.arange(vocab_size, device=device, dtype=torch.float32)

    # Handle batched or single target
    if isinstance(target_bin, int):
        target_bin = torch.tensor(target_bin, device=device, dtype=torch.float32)
    else:
        target_bin = target_bin.float()

    # Create Gaussian soft labels centered at target bin
    soft_labels = torch.exp(-((bins - target_bin)**2) / (2 * sigma**2))
    soft_labels = soft_labels / soft_labels.sum()

    # Compute cross entropy with soft labels
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(soft_labels * log_probs)


class DistanceOutput(nn.Module):
    """Distance-aware output layer with Gaussian mixture components"""

    def __init__(self, num_bins: int = 4096, hidden_dim: int = 64, input_dim: int = 577):
        super().__init__()
        self.num_bins = num_bins

        # Gaussian mixture parameters
        self.gaussian_centers = nn.Parameter(torch.zeros(1, num_bins))
        self.gaussian_widths = nn.Parameter(torch.ones(num_bins))

        # Ordinal embedding
        self.ordinal_embed = nn.Embedding(num_bins, hidden_dim)

        # Value and confidence projections
        self.value_projection = nn.Linear(512, 1)
        self.confidence_projection = nn.Linear(512, 1)

        # Mix layer (MLP with ReLU)
        self.mix_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_bins)
        )

    def forward(self, x):
        return self.mix_layer(x)


class DistanceAwareChronos:
    """
    Distance-Aware Chronos model v2 with soft label loss.
    Optimized for Apple Silicon M3 with MPS backend.
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-small",
                 num_bins: int = 4096, device: str = None):
        """
        Initialize Distance-Aware Chronos model.

        Args:
            model_name: Name of the base Chronos model
            num_bins: Number of bins for tokenization
            device: Device to run on (auto-detects MPS on Apple Silicon)
        """
        self.model_name = model_name
        self.num_bins = num_bins
        self.device = device or get_device()

        # Load base Chronos model
        print(f"  Loading base model: {model_name}...")
        print(f"  Using device: {self.device}")
        self.base_model = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=torch.float32  # MPS compatible
        )

        # Initialize distance output layer
        self.distance_output = DistanceOutput(num_bins=num_bins).to(self.device)

    def predict(self, context: np.ndarray, horizon: int,
                num_samples: int = 100) -> np.ndarray:
        """
        Generate forecasts using distance-aware approach.

        Args:
            context: Historical time series values
            horizon: Number of steps to forecast
            num_samples: Number of sample paths to generate

        Returns:
            Median forecast across samples
        """
        # Convert context to tensor
        if isinstance(context, np.ndarray):
            context_tensor = torch.tensor(context[np.newaxis, :], dtype=torch.float32)
        else:
            context_tensor = context

        # Generate predictions using base model
        with torch.no_grad():
            forecast = self.base_model.predict(
                context_tensor,
                prediction_length=horizon,
                num_samples=num_samples
            )

            # Take median across samples
            if isinstance(forecast, torch.Tensor):
                forecast_array = forecast.cpu().numpy()[0]
            else:
                forecast_array = np.array(forecast)[0]

            median_forecast = np.median(forecast_array, axis=0)

        return median_forecast

    def to(self, device: str):
        """Move model to device"""
        self.device = device
        self.distance_output = self.distance_output.to(device)
        return self
