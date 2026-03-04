import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

# Load pretrained model from Hugging Face
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Configure the model
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# Run inference
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),       # time series 1
        np.sin(np.linspace(0, 20, 67)), # time series 2
    ],
)

print(point_forecast.shape)    # (2, 12)
print(quantile_forecast.shape) # (2, 12, 10) — mean + 10th to 90th quantiles