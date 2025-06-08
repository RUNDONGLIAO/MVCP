# MVCP - Automatic 5P Change Point Model

<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">

MVCP (Measurement & Validation Change Point Model) is a Python toolkit for analyzing building energy consumption and temperature relationships. Based on the five-parameter (5P) change point model, it automatically selects the best change points and analyzes energy consumption curve characteristics.

## Features

- Automatically selects optimal change points based on data patterns
- Supports 5P model (dual change points), 3P heating model, and 3P cooling model
- Automatically calculates various diagnostic statistics (R², RMSE, CV(RMSE), NMBE, etc.)
- Generates detailed Excel reports
- Provides a simple API requiring just two parameters (temperature and energy)

## Citation

If you use the code derived from it in academic work, please cite:

will be here soon....

## Usage

### Basic Usage

Simply provide temperature and energy data to run the model:

```python
from mvcp import run_mvcp_model

temperature = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35]  # example temperature data
energy = [25, 24, 22, 20, 19, 19, 20, 22, 24, 27, 30]       # example energy consumption data

results = run_mvcp_model(temperature, energy)
```

### Advanced Usage

```python
from mvcp import run_mvcp_model
import pandas as pd

# Method 1: Using data lists directly
temperature = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35]  # example temperature data
energy = [25, 24, 22, 20, 19, 19, 20, 22, 24, 27, 30]       # example energy consumption data

results = run_mvcp_model(temperature, energy)

# Method 2: Importing data from CSV/Excel files
data = pd.read_csv("your_data.csv")
results = run_mvcp_model(
    temperature=data["Temperature"],
    energy=data["Energy"],
    output_file="results/MVCP_analysis.xlsx"
)
```

You can access the underlying model functions for visualization:

```python
from mvcp import five_parameter_model
import numpy as np
import matplotlib.pyplot as plt

temperature = np.array([...])  # your temperature data
energy = np.array([...])       # your energy consumption data

# Run the model and get detailed parameters
params, sse, diagnostics, predictions, x_data, y_data = five_parameter_model(temperature, energy)

# Custom visualization
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label="Actual Data")
plt.plot(x_data, predictions, 'r-', label="Model Prediction")
plt.axvline(x=params[1], color='g', linestyle='--', label="Heating Change Point")
plt.axvline(x=params[4], color='b', linestyle='--', label="Cooling Change Point")
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.title(f"5P Model Analysis (R²={diagnostics['R2']:.4f})")
plt.legend()
plt.grid(True)
plt.show()
```
