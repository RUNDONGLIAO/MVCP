from mvcp import five_parameter_model
from mvcp.simple_interface import run_mvcp_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualize_results(params, diagnostics, predictions, x_data, y_data):
    """Visualize model results based on actual model type used"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label="Actual Data")
    plt.plot(x_data, predictions, 'r-', label="Model Prediction")
    
    # Determine the type of model actually used
    model_type = "5P"  # default
    if np.isclose(params[3], 0):  # Only the heating part (3PH)
        model_type = "3PH"
        plt.axvline(x=params[1], color='g', linestyle='--', label="Heating Change Point")
    elif np.isclose(params[0], 0):  # Only the cooling part (3PC)
        model_type = "3PC"
        plt.axvline(x=params[4], color='b', linestyle='--', label="Cooling Change Point")
    else:  # Complete 5P model
        plt.axvline(x=params[1], color='g', linestyle='--', label="Heating Change Point")
        plt.axvline(x=params[4], color='b', linestyle='--', label="Cooling Change Point")
    
    plt.xlabel("Temperature")
    plt.ylabel("Energy")
    plt.title(f"{model_type} Model Analysis (R²={diagnostics['R2']:.4f})")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Advanced usage example"""
    
    # Method 1: Import data from CSV file
    try:
        # Load data
        data = pd.read_csv('example_data.csv')
        
        # Ensure data contains necessary columns
        if 'Temperature' in data.columns and 'Energy' in data.columns:
            temperature = data['Temperature'].values
            energy = data['Energy'].values
            print("Successfully read data from CSV file")
        else:
            print("Columns 'Temperature' or 'Energy' not found in CSV file")
            raise Exception("Data column names don't match")
    
    except Exception as e:
        print(f"Error importing data from CSV: {str(e)}")
        
        # Method 2: Provide data directly
        print("\nUsing example data:")
        
        # Example data
        temperature = np.array([10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35])  # example temperature data
        energy = np.array([25, 24, 22, 20, 19, 19, 20, 22, 24, 27, 30])       # example energy consumption data
    
    # Run the model and get detailed parameters
    params, sse, diagnostics, predictions, x_data, y_data = five_parameter_model(temperature, energy)
    
    # Visualize results
    visualize_results(params, diagnostics, predictions, x_data, y_data)
    
    # Save results to Excel file (similar to basic_usage.py)
    results = run_mvcp_model(
        temperature=temperature,
        energy=energy,
        output_file="results/MVCP_analysis_example.xlsx"
    )

if __name__ == "__main__":
    main()