### Clone metrics.py file on CSD3 and read it here
### Read density.py file

import importlib
import xarray as xr
import sys
from pathlib import Path

# Define paths to the .nc file and metric file
nc_file_path = "../nemo_4.2.1/tests/DINO/EXP00/DINO_1y_grid_T.nc"  # Replace with your .nc file
metric_file_path = "../Metrics-Ocean/metrics.py"  # Path to your metrics file
density_file_path = 'lib/density.py'

# Import the metrics module dynamically
metric_module = Path(metric_file_path).stem  # Extract module name from file
spec = importlib.util.spec_from_file_location(metric_module, metric_file_path)
metrics = importlib.util.module_from_spec(spec)
sys.modules[metric_module] = metrics
spec.loader.exec_module(metrics)

# Open the .nc file using xarray
try:
    dataset = xr.open_dataset(nc_file_path)
    print(f"Successfully loaded dataset from {nc_file_path}")
except FileNotFoundError:
    print(f"Error: File {nc_file_path} not found.")
    sys.exit(1)
    

# List all variables in the dataset
print("\nAvailable variables in the dataset:")
print(dataset.data_vars)

# Define variable mapping (dataset name -> metric file name)
variable_mapping = {
    "rhop": "density",  
    "deptht": "depth",
}

# Define variables and metrics to apply
variables_to_check = ["rhop"]  # Replace with your variable names
metrics_to_apply = ["check_density"]  # Replace with your metric functions

results = {}
for var in variables_to_check:
    if var in dataset:
        print(f"\nApplying metrics to variable: {var}")
        variable_data = dataset[var]
        metric_var_name = variable_mapping.get(var, var)
        results[var] = {}
        
        # Apply metrics from metrics.py
        for metric in metrics_to_apply:
            if hasattr(metrics, metric):
                metric_function = getattr(metrics, metric)
                try:
                    print(variable_data)
                    print(metric_var_name)
                    print(metric_function)
                    result = metric_function(variable_data, metric_var_name)
                    print(result)
                    results[var][metric] = result
                    print(f"Metric '{metric}' applied to '{metric_var_name}'. Result: {result}")
                except Exception as e:
                    print(f"Error applying metric '{metric}' to '{metric_var_name}': {e}")
            else:
                print(f"Metric '{metric}' not found in {metric_file_path}")