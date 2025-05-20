import numpy as np
import pandas as pd
from .utils import calculate_diagnostics, create_default_diagnostics, calculate_predictions, prepare_model_results

# ------------------------------
# Model Functions
# ------------------------------

def three_parameter_heating(x, y, internal=False):
    """
    Three-parameter heating model:
       y = b0 + b1 * max(b2 - x, 0)
    
    Iterates through candidate change points to find optimal coefficients
    that minimize SSE.
    
    Args:
        x: Independent variable (temperature)
        y: Dependent variable (energy consumption)
        internal: If True, return additional best_pred for internal use
        
    Returns:
        If internal=False: tuple of (best_coeffs, best_sse, diagnostics)
        If internal=True: tuple of (best_coeffs, best_sse, best_pred)
    """
    x_arr = np.array(x).flatten()
    y_arr = np.array(y).flatten()
    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]
    n = len(x_sorted)
    
    best_coeffs = [None, None, None]
    best_sse = np.inf
    best_pred = None
    
    # Iterate through possible change points
    for m in range(3, n + 1):
        lower_count = m - 1
        
        # Calculate b0 (base load)
        b0_candidate = np.mean(y_sorted[lower_count:])
        
        # Calculate statistics for slope determination
        sum_x_lower = np.sum(x_sorted[:lower_count])
        sum_y_lower = np.sum(y_sorted[:lower_count])
        sum_x2_lower = np.sum(x_sorted[:lower_count] ** 2)
        sum_xy_lower = np.sum(x_sorted[:lower_count] * y_sorted[:lower_count])
        
        # Calculate b1 (slope)
        denominator_b1 = lower_count * sum_x2_lower - sum_x_lower ** 2
        if denominator_b1 == 0:
            continue
        b1_candidate = (sum_x_lower * sum_y_lower - lower_count * sum_xy_lower) / denominator_b1
        
        # Handle case where slope is zero (horizontal line)
        if np.isclose(b1_candidate, 0):
            b2_candidate = np.median(x_arr)
            model_pred = b0_candidate * np.ones_like(x_arr)
            sse_candidate = np.sum((y_arr - model_pred) ** 2)
            if sse_candidate < best_sse:
                best_sse = sse_candidate
                best_coeffs = [b0_candidate, 0, b2_candidate]
                best_pred = model_pred
            continue
        
        # Calculate b2 (change point)
        b2_candidate = (sum_y_lower - lower_count * b0_candidate + b1_candidate * sum_x_lower) / (lower_count * b1_candidate)
        
        # Compute predictions and SSE
        model_pred = b0_candidate + b1_candidate * np.maximum(b2_candidate - x_arr, 0)
        sse_candidate = np.sum((y_arr - model_pred) ** 2)
        
        # Update best model if current is better
        if sse_candidate < best_sse:
            best_sse = sse_candidate
            best_coeffs = [b0_candidate, b1_candidate, b2_candidate]
            best_pred = model_pred
            
    # Convert heating slope b1 to its negative value before returning
    if best_coeffs[1] is not None:
        return_coeffs = [best_coeffs[0], -best_coeffs[1], best_coeffs[2]]
    else:
        return_coeffs = best_coeffs
    
    # Handle the return based on internal flag    
    if internal:
        return return_coeffs, best_sse, best_pred
    else:
        diagnostics = calculate_diagnostics(x_arr, y_arr, best_pred, return_coeffs, best_sse, model_type="3PH")
        return return_coeffs, best_sse, diagnostics


def three_parameter_cooling(x, y, internal=False):
    """
    Three-parameter cooling model:
       y = b0 + b1 * max(x - b2, 0)
    
    Iterates through candidate change points to find optimal coefficients
    that minimize SSE.
    
    Args:
        x: Independent variable (temperature)
        y: Dependent variable (energy consumption)
        internal: If True, return additional best_pred for internal use
        
    Returns:
        If internal=False: tuple of (best_coeffs, best_sse, diagnostics)
        If internal=True: tuple of (best_coeffs, best_sse, best_pred)
    """
    x_arr = np.array(x).flatten()
    y_arr = np.array(y).flatten()
    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]
    n = len(x_sorted)
    
    best_coeffs = [None, None, None]
    best_sse = np.inf
    best_pred = None
    
    # Iterate through possible change points
    for m in range(2, n):
        lower_count = m - 1
        upper_count = n - lower_count
        
        # Calculate b0 (base load)
        b0_candidate = np.mean(y_sorted[:lower_count])
        
        # Data for the upper segment (above change point)
        x_upper = x_sorted[lower_count:]
        y_upper = y_sorted[lower_count:]
        
        # Calculate statistics for slope determination
        sum_x_upper = np.sum(x_upper)
        sum_y_upper = np.sum(y_upper)
        sum_x2_upper = np.sum(x_upper ** 2)
        sum_xy_upper = np.sum(x_upper * y_upper)
        
        # Calculate b1 (slope)
        denominator_b1 = upper_count * sum_x2_upper - sum_x_upper ** 2
        if denominator_b1 == 0:
            continue
        b1_candidate = (upper_count * sum_xy_upper - sum_x_upper * sum_y_upper) / denominator_b1
        
        # Handle case where slope is zero (horizontal line)
        if np.isclose(b1_candidate, 0):
            b2_candidate = np.median(x_arr)
            model_pred = b0_candidate * np.ones_like(x_arr)
            sse_candidate = np.sum((y_arr - model_pred) ** 2)
            if sse_candidate < best_sse:
                best_sse = sse_candidate
                best_coeffs = [b0_candidate, 0, b2_candidate]
                best_pred = model_pred
            continue
        
        # Calculate b2 (change point)
        b2_candidate = np.sum(b0_candidate + b1_candidate * x_upper - y_upper) / (upper_count * b1_candidate)
        
        # Compute predictions and SSE
        model_pred = b0_candidate + b1_candidate * np.maximum(x_arr - b2_candidate, 0)
        sse_candidate = np.sum((y_arr - model_pred) ** 2)
        
        # Update best model if current is better
        if sse_candidate < best_sse:
            best_sse = sse_candidate
            best_coeffs = [b0_candidate, b1_candidate, b2_candidate]
            best_pred = model_pred
    
    # Handle the return based on internal flag
    if internal:
        return best_coeffs, best_sse, best_pred
    else:        
        diagnostics = calculate_diagnostics(x_arr, y_arr, best_pred, best_coeffs, best_sse, model_type="3PC")
        return best_coeffs, best_sse, diagnostics


def five_parameter_model(x, y):
    """
    Five-parameter model implementation:
    
    1. If y values are constant, returns a constant model
    2. Otherwise, fits a quadratic curve to find minimum point x_min
    3. Segments data at x_min:
       - For x <= x_min: applies three-parameter heating model
       - For x > x_min: applies three-parameter cooling model
       - Ensures continuity by averaging the intercepts
    4. Builds overall prediction, calculates SSE and diagnostics
    
    Args:
        x: Independent variable (temperature)
        y: Dependent variable (energy consumption)
        
    Returns:
        tuple: (params, overall_sse, diagnostics, predictions, x_array, y_array)
        where params = [heating_slope, heating_change_point, b0, cooling_slope, cooling_change_point]
    """
    x_arr = np.array(x).flatten()
    y_arr = np.array(y).flatten()
    
    # Step 1: Check if constant model applies using interval analysis
    # Sort data by temperature for interval creation
    sorted_indices = np.argsort(x_arr)
    x_sorted = x_arr[sorted_indices]
    y_sorted = y_arr[sorted_indices]
    
    # Calculate overall statistics for comparison
    overall_mean = np.mean(y_arr)
    overall_median = np.median(y_arr)
    
    # Split into 4 temperature intervals
    n_intervals = 4
    interval_size = max(1, len(x_arr) // n_intervals)
    means = []
    medians = []
    
    # Calculate mean and median for each interval
    for i in range(n_intervals):
        start_idx = i * interval_size
        end_idx = (i + 1) * interval_size if i < n_intervals - 1 else len(y_sorted)
        if start_idx >= end_idx:  # Ensure valid interval
            continue
        interval_y = y_sorted[start_idx:end_idx]
        means.append(np.mean(interval_y))
        medians.append(np.median(interval_y))
    
    # If we have enough intervals for comparison
    if len(means) >= 2:
        # Find max and min values
        max_mean = max(means)
        min_mean = min(means)
        max_median = max(medians)
        min_median = min(medians)
        
        # Calculate relative differences (as percentages)
        mean_max_diff = abs((max_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else float('inf')
        mean_min_diff = abs((min_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else float('inf')
        median_max_diff = abs((max_median - overall_median) / overall_median) * 100 if overall_median != 0 else float('inf')
        median_min_diff = abs((min_median - overall_median) / overall_median) * 100 if overall_median != 0 else float('inf')
        
        # Check if constant model applies (within ±5% threshold)
        threshold = 5.0  # 5% threshold for variation
        if (mean_max_diff <= threshold and mean_min_diff <= threshold and 
            median_max_diff <= threshold and median_min_diff <= threshold):
            # Fit a simple linear regression model using least squares
            X_design_linear = np.vstack([np.ones(len(x_arr)), x_arr]).T
            linear_beta = np.linalg.lstsq(X_design_linear, y_arr, rcond=None)[0]
            intercept, slope = linear_beta
            
            # Calculate predictions using linear model
            pred_linear = intercept + slope * x_arr
            residuals = y_arr - pred_linear
            overall_sse = np.sum(residuals**2)
            
            # Calculate R² for linear model
            mean_y = np.mean(y_arr)
            sst = np.sum((y_arr - mean_y) ** 2)
            r2 = 1 - overall_sse / sst if sst != 0 else 0.0
            
            # Set parameters to match 5P model structure
            median_x = np.median(x_arr)
            params = [slope, median_x, intercept, slope, median_x]
            
            # Update diagnostics for linear model
            n = len(y_arr)
            df_error = n - 2  # Linear model has 2 parameters
            rmse = np.sqrt(overall_sse / df_error) if df_error > 0 else 0.0
            
            diagnostics = {
                "N": n,
                "R2": r2,
                "AdjR2": 1 - (1 - r2) * ((n - 1) / df_error) if df_error > 0 and n > 2 else 0.0,
                "RMSE": rmse,
                "CV(RMSE)": (rmse / abs(mean_y)) * 100 if mean_y != 0 else 0.0,
                "NMBE": (np.mean(pred_linear - y_arr) / mean_y) * 100 if mean_y != 0 else 0.0,
                "p_value": 0.0 if r2 > 0 else 1.0,
                "DW": np.sum(np.diff(residuals)**2) / overall_sse if overall_sse != 0 else 0.0,
            }
            return params, overall_sse, diagnostics, pred_linear, x_arr, y_arr
    
    # Step 2: Quadratic fit to find minimum point
    X_design = np.vstack([np.ones(len(x_arr)), x_arr, x_arr**2]).T
    beta = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y_arr)
    b0_poly, b1_poly, b2_poly = beta
    
    # Find minimum/maximum point of the quadratic curve
    if np.isclose(b2_poly, 0):
        x_min = np.median(x_arr)
    else:
        x_min = -b1_poly / (2 * b2_poly)
    
    # Check if x_min is within the data range
    x_min_in_range = (x_min >= np.min(x_arr)) and (x_min <= np.max(x_arr))
    
    # If x_min is outside data range, select single model based on linear regression slope
    if not x_min_in_range:
        # Fit simple linear regression model
        X_design_linear = np.vstack([np.ones(len(x_arr)), x_arr]).T
        linear_beta = np.linalg.lstsq(X_design_linear, y_arr, rcond=None)[0]
        intercept, slope = linear_beta
        
        if slope <= 0:  # Negative slope, use 3PH model (heating mode)
            # x_min is to the right of data, data shows decreasing trend, use 3PH
            heating_params, sse_heating, pred_heating = three_parameter_heating(x_arr, y_arr, internal=True)
            if heating_params[0] is None:
                default_diag = create_default_diagnostics(len(y_arr))
                return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan, default_diag, None, x_arr, y_arr
            
            # Construct a parameter format compatible with 5P: [b1_heating, b2_heating, b0, b1_cooling, b2_cooling]
            b0, b1, b2 = heating_params
            # Use heating parameters
            b1_heating = b1
            b2_heating = b2
            # Set the cooling parameters to zero or irrelevant values
            b1_cooling = 0
            b2_cooling = 0
            b0_final = b0
            
            # Calculate the predicted values and residuals
            pred = calculate_predictions("3PH", x_arr, heating_params)
            residuals = y_arr - pred
            overall_sse = np.sum(residuals**2)
            
            # Return parameters and diagnostic information
            params = [b1_heating, b2_heating, b0_final, b1_cooling, b2_cooling]
            diagnostics = calculate_diagnostics(x_arr, y_arr, pred, params, overall_sse, model_type="3PH")
            
            return params, overall_sse, diagnostics, pred, x_arr, y_arr
            
        else:  # Positive slope, use 3PC model (cooling mode)
            # x_min is to the left of data, data shows increasing trend, use 3PC
            cooling_params, sse_cooling, pred_cooling = three_parameter_cooling(x_arr, y_arr, internal=True)
            if cooling_params[0] is None:
                default_diag = create_default_diagnostics(len(y_arr))
                return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan, default_diag, None, x_arr, y_arr
            
            # Construct a parameter format compatible with 5P:[b1_heating, b2_heating, b0, b1_cooling, b2_cooling]
            b0, b1, b2 = cooling_params
            # Set the heating parameters to zero or irrelevant values
            b1_heating = 0
            b2_heating = 0
            # Use cooling parameters
            b1_cooling = b1
            b2_cooling = b2
            b0_final = b0
            
            # Calculate the predicted values and residuals
            pred = calculate_predictions("3PC", x_arr, cooling_params)
            residuals = y_arr - pred
            overall_sse = np.sum(residuals**2)
            
            # Return parameters and diagnostic information
            params = [b1_heating, b2_heating, b0_final, b1_cooling, b2_cooling]
            diagnostics = calculate_diagnostics(x_arr, y_arr, pred, params, overall_sse, model_type="3PC")
            
            return params, overall_sse, diagnostics, pred, x_arr, y_arr
    
    # Step 3: Segment data, ensure each segment has enough points
    left_mask = x_arr < x_min
    right_mask = x_arr > x_min
    if np.sum(left_mask) < 3 or np.sum(right_mask) < 3:
        # Try median as change point
        x_min = np.median(x_arr)
        left_mask = x_arr < x_min
        right_mask = x_arr > x_min
        
        # If still insufficient points on either side, use single model
        if np.sum(left_mask) < 3 or np.sum(right_mask) < 3:
            X_design_linear = np.vstack([np.ones(len(x_arr)), x_arr]).T
            linear_beta = np.linalg.lstsq(X_design_linear, y_arr, rcond=None)[0]
            intercept, slope = linear_beta

            if slope <= 0:  # Negative slope, use 3PH model
                heating_params, sse_heating, pred_heating = three_parameter_heating(x_arr, y_arr, internal=True)
                if heating_params[0] is None:
                    default_diag = create_default_diagnostics(len(y_arr))
                    return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan, default_diag, None, x_arr, y_arr
                
                # Construct parameters in 5P format: [b1_heating, b2_heating, b0, b1_cooling, b2_cooling]
                b0, b1, b2 = heating_params
                b1_heating = b1
                b2_heating = b2
                b1_cooling = 0  # No cooling effect
                b2_cooling = np.max(x_arr)  # Set to max temperature
                b0_final = b0
                
                # Calculate predictions and metrics
                pred = calculate_predictions("3PH", x_arr, heating_params)
                residuals = y_arr - pred
                overall_sse = np.sum(residuals**2)
                
                # Return parameters and diagnostic information
                params = [b1_heating, b2_heating, b0_final, b1_cooling, b2_cooling]
                diagnostics = calculate_diagnostics(x_arr, y_arr, pred, params, overall_sse, model_type="3PH")
                
                return params, overall_sse, diagnostics, pred, x_arr, y_arr
            else:
                # More points on right side, use 3PC model
                cooling_params, sse_cooling, pred_cooling = three_parameter_cooling(x_arr, y_arr, internal=True)
                if cooling_params[0] is None:
                    default_diag = create_default_diagnostics(len(y_arr))
                    return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan, default_diag, None, x_arr, y_arr
                
                # Construct parameters in 5P format: [b1_heating, b2_heating, b0, b1_cooling, b2_cooling]
                b0, b1, b2 = cooling_params
                b1_heating = 0  # No heating effect
                b2_heating = np.min(x_arr)  # Set to min temperature
                b1_cooling = b1
                b2_cooling = b2
                b0_final = b0
                
                # Calculate predictions and metrics
                pred = calculate_predictions("3PC", x_arr, cooling_params)
                residuals = y_arr - pred
                overall_sse = np.sum(residuals**2)
                
                # Return parameters and diagnostic information
                params = [b1_heating, b2_heating, b0_final, b1_cooling, b2_cooling]
                diagnostics = calculate_diagnostics(x_arr, y_arr, pred, params, overall_sse, model_type="3PC")
                
                return params, overall_sse, diagnostics, pred, x_arr, y_arr
    
    # Fit separate models to each segment
    heating_params, sse_heating, pred_heating = three_parameter_heating(x_arr[left_mask], y_arr[left_mask], internal=True)
    cooling_params, sse_cooling, pred_cooling = three_parameter_cooling(x_arr[right_mask], y_arr[right_mask], internal=True)
    
    # Check for valid solutions
    if heating_params[0] is None or cooling_params[0] is None:
        default_diag = create_default_diagnostics(len(y_arr))
        return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan, default_diag, None, x_arr, y_arr
    
    b0_heating, b1_heating, b2_heating = heating_params
    b0_cooling, b1_cooling, b2_cooling = cooling_params
    
    # Step 4: Ensure continuity by averaging intercepts
    b0_final = (b0_heating + b0_cooling) / 2.0
    
    # Use helper function to calculate predictions
    params = [b1_heating, b2_heating, b0_final, b1_cooling, b2_cooling]
    pred = calculate_predictions("5P", x_arr, params)
    
    # Directly calculate residuals and SSE without cumulative values
    residuals = y_arr - pred
    overall_sse = np.sum(residuals**2)
    
    # Calculate diagnostic metrics
    diagnostics = calculate_diagnostics(x_arr, y_arr, pred, params, overall_sse, model_type="5P")
    
    return params, overall_sse, diagnostics, pred, x_arr, y_arr