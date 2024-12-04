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


# Example to demonstrate the use of the simulate_data function, using the define_dataframe_structure output as input
data = [{
    'name':'age', 'reps': [25, 30, 35]},
    {'name': 'salary', 'reps': [50000, 60000, 70000]
}]

seed_df = cm.define_dataframe_structure(data)

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

# Example to demonstrate the use of the export_to_csv function
cm.export_to_csv(new_df, "simulated_data.csv", delimiter=",", include_index=False)

# Example to demonstrate the use of the export_formatted function
cm.export_formatted(new_df, "simulated_data.txt")



# Example to demonstrate the use of the non_globular_cluster function
# Example usage
seed_df = cm.define_dataframe_structure([
    {'name': 'x', 'reps': [1, 2, 3]},
    {'name': 'y', 'reps': [4, 5, 6]}
])
col_specs = {
    'x': {'distribution': 'normal', 'variance': 1.0},
    'y': {'distribution': 'uniform', 'variance': 2.0}
}
n_points = 100
simulated_df = cm.non_globular_cluster(seed_df, n_points, col_specs, random_state=42)
print(simulated_df)

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(simulated_df)

plt.scatter(simulated_df['x'], simulated_df['y'], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()