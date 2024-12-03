###
## cluster_maker
## A package to simulate clusters of data points.
## J. Foadi - University of Bath - 2024
##
## Module data_exporter
###

import pandas as pd

## Function to export to CSV
def export_to_csv(data, filename, delimiter=",", include_index=False):
    """
    Export the DataFrame to a CSV file.

    Parameters:
        data (pd.DataFrame): The DataFrame to export.
        filename (str): Name of the output CSV file.
        delimiter (str): Delimiter for the CSV file (default: ',').
        include_index (bool): Whether to include the DataFrame index (default: False).

    Returns:
        None
    """
    try:
        data.to_csv(filename, sep=delimiter, index=include_index)
        print(f"Data successfully exported to {filename}")
    except Exception as e:
        print(f"Error exporting data to CSV: {e}")

    return None