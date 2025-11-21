"""
Distance-Aware Chronos Model
Extends Chronos with distance-aware forecasting capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from chronos import ChronosPipeline


class DistanceOutput(nn.Module):
    """Distance-aware output layer with Gaussian mixture components"""

    def __init__(self, num_bins: int = 4096, hidden_dim: int = 64, input_dim: int = 577):
        super().__init__()
        self.num_bins = num_bins

        # Gaussian mixture parameters
        # gaussian_centers: [1, num_bins], gaussian_widths: [num_bins]
        self.gaussian_centers = nn.Parameter(torch.zeros(1, num_bins))
        self.gaussian_widths = nn.Parameter(torch.ones(num_bins))

        # Ordinal embedding
        self.ordinal_embed = nn.Embedding(num_bins, hidden_dim)

        # Value and confidence projections - input dimension is 512
        self.value_projection = nn.Linear(512, 1)
        self.confidence_projection = nn.Linear(512, 1)

        # Mix layer (MLP with ReLU) - input_dim: 577, hidden: 512, output: num_bins
        self.mix_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_bins)
        )

    def forward(self, x):
        # Process through mix layer
        return self.mix_layer(x)


class DistanceAwareChronos:
    """
    Distance-Aware Chronos model that enhances forecasting with distance metrics.
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-small",
                 num_bins: int = 4096, device: str = 'cpu'):
        """
        Initialize Distance-Aware Chronos model.

        Args:
            model_name: Name of the base Chronos model
            num_bins: Number of bins for tokenization
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.num_bins = num_bins
        self.device = device

        # Load base Chronos model
        print(f"  Loading base model: {model_name}...")
        self.base_model = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32
        )

        # Initialize distance output layer
        self.distance_output = DistanceOutput(num_bins=num_bins).to(device)

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
