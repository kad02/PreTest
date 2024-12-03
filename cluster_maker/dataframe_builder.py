###
## cluster_maker
## A package to simulate clusters of data points.
## J. Foadi - University of Bath - 2024
##
## Module dataframe_builder
###

## Libraries needed
import pandas as pd
import numpy as np

## Function to define the wanted data structure
def define_dataframe_structure(column_specs):
    """
    Finds the longest column
    and extends all columns to match that length with NaN values appended to the end.

    Parameters:
        column_specs (list of dict): List of dictionaries with column specifications.
    
    Returns:
        pd.DataFrame: DataFrame with extended columns.
    """
    # Prepare data dictionary
    data = {}
    max_length = 0

    # Find the maximum length of representative points
    for spec in column_specs:
        max_length = max(max_length, len(spec.get('reps', [])))

    for spec in column_specs:
        name = spec['name']
        reps = spec.get('reps', [])
        # Extend numerical columns with NaN to match max_length
        extended_points = reps + [np.nan] * (max_length - len(reps))
        data[name] = extended_points

    return pd.DataFrame(data)

## Function to simulate data
def simulate_data(seed_df, n_points=100, col_specs=None, random_state=None):
    """
    Simulate data points based on the seed DataFrame.
    Each row of the seed DataFrame has a "distribution" and "variance" specification for each column.
    "distribution" can be 'normal' or 'uniform', default is 'normal'.
    "variance" is the standard deviation for 'normal' and half the range for 'uniform', default is 1.
    Add extra column to the seed DataFrame containing a random sample from the given distribution with the given variance.

    Parameters:
        seed_df (pd.DataFrame): The seed DataFrame to simulate data from.
        n_points (int): Number of points to simulate for each representative point (default: 100).
        col_specs (dict): Dictionary with column specifications (default: None).
        random_state (int): Random state for reproducibility (default: None).
    
    Returns:   
        pd.DataFrame: Simulated data points.
    
    
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    simulated_data = []

    for _, representative in seed_df.iterrows():
        for _ in range(n_points):
            simulated_point = {}
            for col in seed_df.columns:
                # Numerical columns: apply column-specific specifications
                if col_specs and col in col_specs:
                    dist = col_specs[col].get('distribution', 'normal')
                    variance = col_specs[col].get('variance', 1.0)

                    if dist == 'normal':
                        simulated_point[col] = representative[col] + np.random.normal(0, np.sqrt(variance))
                    elif dist == 'uniform':
                        simulated_point[col] = representative[col] + np.random.uniform(-variance, variance)
                    else:
                        raise ValueError(f"Unsupported distribution: {dist}")
                else:
                    raise ValueError(f"Column {col} has no specifications in col_specs.")
            simulated_data.append(simulated_point)
    
    return pd.DataFrame(simulated_data)
