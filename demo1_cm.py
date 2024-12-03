## Import cluster_maker
import cluster_maker as cm
import pandas as pd
import numpy as np

# Example to demonstrate the use of the define_dataframe_structure function
# Notice how age has 3 representative points, salary has 2, and department has 4
# The empty spaces should be filled with NaN values
column_specs = [
    {
        'name': 'age',
        'reps': [25, 30, 35]
    },
    {
        'name': 'salary',
        'reps': [50000, 60000]
    },
    {
        'name': 'department',
        'reps': ['HR', 'Engineering', 'Marketing', 'Sales']
    }
]

# Generate the DataFrame
df = cm.define_dataframe_structure(column_specs)
print(df)

print("\n\n\n")


# Example to demonstrate the use of the simulate_data function
data = {
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}
seed_df = pd.DataFrame(data)

# Example col_specs
col_specs = {
    'age': {
        'distribution': 'normal',
        'variance': 4.0
    },
    'salary': {
        'distribution': 'uniform',
        'variance': 5000
    }
}

new_df = cm.simulate_data(seed_df, n_points=10, col_specs=col_specs)

print(new_df)