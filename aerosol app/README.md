# Aerosol App

This folder contains the aerosol size-distribution workflow.

## Files

- `data/`  
  Contains the input Excel file and the processed CSV file.

- `codes/`  
  Contains the Python scripts used for data processing and modelling.

- `outputs/`  
  Contains the generated figures and results.

## Flow

1. Extract daily lognormal parameters from the SMPS/CPC Excel file.
2. Save the daily parameters as `lognormal_parameters_2022.csv`.
3. Run the Fisher–Rao forecasting script.
4. Save the generated plots in the `outputs/` folder.
