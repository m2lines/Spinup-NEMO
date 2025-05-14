**Overview**

This project provides a flexible framework for oceanographic time‐series forecasting. It separates dimensionality reduction (DR) and forecasting into interchangeable components, enabling you to swap in your own algorithms with minimal changes.

---

## 1. Installation

1. **Clone the repository**

   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```
2. **Set up a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 2. Project Structure

```
├── lib/
│   ├── dimensionality_reduction.py   # Base + PCA/KPCA classes
│   ├── forecast_technique.py         # Base + GP/Recursive forecasters
│   └── restart.py                    # Restart-file utilities
├── main_forecast.py                  # CLI for preparing & forecasting
├── main_restart.py                   # CLI for updating restart files
├── forecast.py                       # Orchestrates DR + forecasting
├── techniques_config.yaml            # Select DR & forecast technique
├── ocean_terms.yaml                  # Map variable names in NEMO grids
└── jumper.ipynb                      # Example notebook using PCA
```

---

## 3. Configuration

All user‐selectable techniques live in `techniques_config.yaml`:

```yaml
DR_technique:
  name: PCA                # Options: PCA, KernelPCA, or your custom class
Forecast_technique:
  name: GaussianProcessRecursiveForecaster  # Options: GaussianProcessForecaster, GaussianProcessRecursiveForecaster, or your custom class
```

Maps for NEMO grids live in `ocean_terms.yaml`:

```yaml
Terms:
  Salinity: soce
  Temperature: toce
  SSH: ssh
```

---

## 4. Adding a Custom Dimensionality Reduction

1. **Create a subclass of** `DimensionalityReduction` in `lib/dimensionality_reduction.py`:

   ```python
   from lib.dimensionality_reduction import DimensionalityReduction

   class MyDR(DimensionalityReduction):
       def __init__(self, comp, **kwargs):
           self.comp = comp
           # initialize other parameters

       def set_from_simulation(self, sim):
           # copy metadata from Simulation
           ...

       def decompose(self, simulation, length):
           # return components, model-instance, mask
           ...

       @staticmethod
       def reconstruct_predictions(predictions, n, info, begin=0):
           # return mask, reconstructed-array
           ...

       def reconstruct_components(self, n):
           # optional: full reconstruction
           ...

       def get_component(self, n):
           # optional: single component map
           ...

       def error(self, n):
           # optional: reconstruction error
           ...
   ```
2. **Register** your class in `forecast.py`:

   ```python
   from lib.dimensionality_reduction import MyDR

   Dimensionality_reduction_techniques = {
       "PCA": DimensionalityReductionPCA,
       "KernelPCA": DimensionalityReductionKernelPCA,
       "MyDR": MyDR,
   }
   ```
3. **Select** your DR in `techniques_config.yaml`:

   ```yaml
   DR_technique:
     name: MyDR
   ```

---

## 5. Adding a Custom Forecasting Technique

1. **Create a subclass of** `BaseForecaster` in `lib/forecast_technique.py`:

   ```python
   from lib.forecast_technique import BaseForecaster

   class MyForecaster(BaseForecaster):
       def __init__(self, **params):
           # initialize your model
           ...

       def apply_forecast(self, y_train, x_train, x_pred):
           # fit your model on y_train (and x_train), predict x_pred
           # return (y_hat, y_hat_std)
           ...
   ```
2. **Instantiate** and register your forecaster in the same file or in `forecast.py`:

   ```python
   from lib.forecast_technique import MyForecaster

   Forecast_techniques = {
       "GaussianProcessForecaster": GaussianProcessForecaster,
       "GaussianProcessRecursiveForecaster": GaussianProcessRecursiveForecaster,
       "MyForecaster": MyForecaster,
   }
   ```
3. **Select** your forecaster in `techniques_config.yaml`:

   ```yaml
   Forecast_technique:
     name: MyForecaster
   ```

---

## 6. Running a Forecast

1. **Prepare & Forecast** via the CLI:

   ```bash
   python main_forecast.py \
     --path /path/to/simulation/files \
     --ye True \
     --start 25 --end 65 \
     --comp 0.9 --steps 30
   ```
2. **Output**:

   * Prepared data in `simu_prepared/{term}/`
   * Forecasted components in `simu_predicted/{term}.npy`

---

## 7. Example Notebook

* `Jumper.ipynb` demonstrates forecasting with `PCA` only. Copy, modify, or extend to test your own techniques.

---
