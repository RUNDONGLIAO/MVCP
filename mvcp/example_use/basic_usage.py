import pandas as pd
from mvcp.simple_interface import run_mvcp_model

def main():
    """Basic Usage Example"""
    
    # Method 1: Import data from CSV file
    try:
        # Load data
        data = pd.read_csv('example_data.csv')
        
        # Ensure data contains necessary columns
        if 'Temperature' in data.columns and 'Energy' in data.columns:
            # Run model
            results = run_mvcp_model(
                temperature=data['Temperature'],
                energy=data['Energy'],
                output_file="results/MVCP_analysis_example.xlsx"
            )
        else:
            print("Columns 'Temperature' or 'Energy' not found in CSV file")
    
    except Exception as e:
        print(f"Error importing data from CSV: {str(e)}")
        
        # Method 2: Provide data directly
        print("\nUsing example data:")
        
        # Example data
        temperature = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35]
        energy = [25, 24, 22, 20, 19, 19, 20, 22, 24, 27, 30]
        
        # Run model
        results = run_mvcp_model(
            temperature=temperature,
            energy=energy,
            output_file="results/MVCP_analysis_example.xlsx"
        )

if __name__ == "__main__":
    main()