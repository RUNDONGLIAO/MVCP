import numpy as np
import pandas as pd
from scipy.stats import f

def calculate_diagnostics(x, y, pred, coeffs, sse, model_type="5P"):
    """
    Calculate model diagnostic metrics for energy models.
    
    Args:
        x: Independent variable (temperature)
        y: Dependent variable (energy consumption)
        pred: Model predictions
        coeffs: Model coefficients
        sse: Sum of squared errors
        model_type: Type of model (default "5P")
        
    Returns:
        Dictionary containing diagnostic metrics
    """
    # If pred is None, return NaN for all metrics
    if pred is None:
        return create_default_diagnostics(len(y))
    
    n = len(y)
    mean_y = np.mean(y)
    residuals = y - pred
    
    # Total sum of squares (SST)
    sst = np.sum((y - mean_y) ** 2)
    
    # Always recalculate SSE to ensure numerical stability
    sse_recalculated = np.sum(residuals**2)
    
    # Use recalculated SSE to compute R²
    r2 = 1 - sse_recalculated / sst if sst != 0 else np.nan
    
    # Model parameter count
    if model_type == "5P":
        params_count = 5
    else:
        params_count = 3
    
    # Degrees of freedom
    df_error = n - params_count
    
    # Root mean square error (RMSE)
    rmse = np.sqrt(sse_recalculated / df_error) if df_error > 0 else np.nan
    
    # Adjusted coefficient of determination
    adj_r2 = 1 - (1 - r2) * ((n - 1) / df_error) if df_error > 0 and n > params_count else np.nan
    
    # Coefficient of variation (CV-RMSE)
    cv_rmse = (rmse / abs(mean_y)) * 100 if mean_y != 0 else np.nan
    
    # Normalized mean bias error (NMBE)
    nmbe = (np.mean(pred - y) / mean_y) * 100 if mean_y != 0 else np.nan
    
    # F-test and p-value
    if df_error > 0 and (1 - r2) != 0:
        f_stat = (r2 / (params_count - 1)) / ((1 - r2) / df_error)
        p_value = f.sf(f_stat, params_count - 1, df_error)
    else:
        p_value = np.nan
    
    # Durbin-Watson statistic (detect autocorrelation in residuals)
    if n > 1 and sse_recalculated != 0:
        dw = np.sum(np.diff(residuals)**2) / sse_recalculated
    else:
        dw = np.nan
    
    diagnostics = {
        "N": n,                 # Sample size
        "R2": r2,               # Coefficient of determination
        "AdjR2": adj_r2,        # Adjusted coefficient of determination
        "RMSE": rmse,           # Root mean square error
        "CV(RMSE)": cv_rmse,    # Coefficient of variation
        "NMBE": nmbe,           # Normalized mean bias error
        "p_value": p_value,     # p-value
        "DW": dw,               # Durbin-Watson statistic
    }
    
    return diagnostics


def create_default_diagnostics(n):
    """Create default diagnostics dictionary with NaN values"""
    return {
        "N": n,
        "R2": np.nan,
        "AdjR2": np.nan,
        "RMSE": np.nan,
        "CV(RMSE)": np.nan,
        "NMBE": np.nan,
        "p_value": np.nan,
        "DW": np.nan,
    }


def calculate_predictions(model_type, x_data, coeffs):
    """Calculate predictions based on model type and coefficients"""
    if model_type == "3PC":
        return coeffs[0] + coeffs[1] * np.maximum(x_data - coeffs[2], 0)
    elif model_type == "3PH":
        return coeffs[0] + (-coeffs[1]) * np.maximum(coeffs[2] - x_data, 0)
    elif model_type == "5P":
        b1_heating, b2_heating, b0, b1_cooling, b2_cooling = coeffs
        pred = np.empty_like(x_data, dtype=float)
        for idx, xi in enumerate(x_data):
            if xi <= b2_heating:
                pred[idx] = b0 + (-b1_heating) * (b2_heating - xi)
            elif xi >= b2_cooling:
                pred[idx] = b0 + b1_cooling * (xi - b2_cooling)
            else:
                pred[idx] = b0
        return pred
    return None


def prepare_model_results(test_case, model_type, coeffs, sse, diagnostics, x_data, y_data, pred):
    """Prepare model results dictionary and details dataframe"""
    details_df = pd.DataFrame({'x': x_data, 'y': y_data})
    
    if "3PC" in model_type:
        result_row = {
            "test_case": test_case,
            "model": model_type,
            "b0": coeffs[0],
            "b1": coeffs[1],
            "b2": coeffs[2],
            "SSE": sse
        }
        if coeffs[0] is not None:
            details_df['prediction'] = calculate_predictions("3PC", details_df['x'], coeffs)
    
    elif "3PH" in model_type:
        result_row = {
            "test_case": test_case,
            "model": model_type,
            "b0": coeffs[0],
            "b1": coeffs[1],
            "b2": coeffs[2],
            "SSE": sse
        }
        if coeffs[0] is not None:
            details_df['prediction'] = calculate_predictions("3PH", details_df['x'], coeffs)
    
    elif "5P" in model_type:
        result_row = {
            "test_case": test_case,
            "model": model_type,
            "Heating slope": coeffs[0],
            "Heating change point": coeffs[1],
            "b0": coeffs[2],
            "Cooling slope": coeffs[3],
            "Cooling change point": coeffs[4],
            "SSE": sse
        }
        if not np.isnan(coeffs[0]):
            details_df['prediction'] = pred
    
    # Add residuals if predictions exist
    if 'prediction' in details_df.columns:
        details_df['residual'] = details_df['y'] - details_df['prediction']
    
    # Add diagnostics to result row
    result_row.update(diagnostics)
    
    return result_row, details_df