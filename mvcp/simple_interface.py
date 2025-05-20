import pandas as pd
import numpy as np
import os
from .core import five_parameter_model
from .utils import prepare_model_results

def run_mvcp_model(temperature, energy, output_file="model_results.xlsx"):
    """
    Run MVCP model and analyze the relationship between temperature and energy consumption data.
    
    Parameters:
        temperature: Temperature data list or array
        energy: Energy consumption data list or array
        output_file: Excel filename for output results
        
    Returns:
        DataFrame: Results containing model parameters and diagnostic metrics
    """
    # Ensure input is numpy array
    x_data = np.array(temperature).flatten()
    y_data = np.array(energy).flatten()
    
    # Validate input data
    if len(x_data) != len(y_data):
        raise ValueError("Temperature and energy data must have the same length")
    
    if len(x_data) < 3:
        raise ValueError("Insufficient data points, recommend at least 3 data points")
    
    # Run 5P model analysis
    print("Analyzing data...")
    params, overall_sse, diags, pred, x_arr, y_arr = five_parameter_model(x_data, y_data)
    model_used = "5P (Five-parameter model)"
    
    # Prepare results
    result_row, details_df = prepare_model_results("MVCP_Analysis", model_used, params, 
                                                overall_sse, diags, x_arr, y_arr, pred)
    
    # Create results DataFrame
    results_df = pd.DataFrame([result_row])
    
    # Create output directory (if it doesn't exist)
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save results
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name="Results", index=False)
        details_df.to_excel(writer, sheet_name="Details", index=False)
    
    print(f"Analysis complete, results saved to {output_file}")
    
    # Print results summary
    print("\nModel Results Summary:")
    if not np.isnan(params[0]):
        # Detect the type of model actually used
        if np.isclose(params[3], 0):  # Only 3PH model (only heating part)
            print("Model Type: 3PH (Three-parameter heating model)")
            print(f"Heating Slope: {params[0]:.4f}")
            print(f"Heating Change Point: {params[1]:.2f}")
            print(f"Base Load (b0): {params[2]:.4f}")
        elif np.isclose(params[0], 0):  # Only 3PC model (only the cooling part)
            print("Model Type: 3PC (Three-parameter cooling model)")
            print(f"Base Load (b0): {params[2]:.4f}")
            print(f"Cooling Slope: {params[3]:.4f}")
            print(f"Cooling Change Point: {params[4]:.2f}")
        else:  # The complete 5P model
            print("Model Type: 5P (Five-parameter model)")
            print(f"Heating Slope: {params[0]:.4f}")
            print(f"Heating Change Point: {params[1]:.2f}")
            print(f"Base Load (b0): {params[2]:.4f}")
            print(f"Cooling Slope: {params[3]:.4f}")
            print(f"Cooling Change Point: {params[4]:.2f}")
    else:
        print("Model fitting failed, please check your data")
    
    print(f"\nModel Diagnostic Metrics:")
    print(f"Sample Size: {diags['N']}")
    print(f"Coefficient of Determination (R²): {diags['R2']:.4f}")
    print(f"Adjusted R-squared: {diags['AdjR2']:.4f}")
    print(f"Root Mean Square Error (RMSE): {diags['RMSE']:.4f}")
    print(f"Coefficient of Variation CV(RMSE): {diags['CV(RMSE)']:.2f}%")
    
    return results_df