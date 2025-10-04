import numpy as np
import pandas as pd
import json
from network_topology import NetworkTopology

def calculate_cuality_matrix(modules, snr_db_values):
    topology = NetworkTopology(modules)
    topology.create_graph()
    connections = topology.get_connections()
    
    quality_matrix = pd.DataFrame(index=modules.keys(), columns=modules.keys())
    
    for mod_a, mod_b, _ in connections:
        quality_metrics = {}
        for snr_db in snr_db_values:
            results = topology.calculate_link_quality(mod_a, mod_b, snr_db)
            quality_metrics[snr_db] = results
        quality_matrix.at[mod_a, mod_b] = quality_metrics
        quality_matrix.at[mod_b, mod_a] = quality_metrics  # Assuming symmetric links
    
    return quality_matrix

def calculate_per(ber):
    if ber < 0 or ber > 1:
        raise ValueError("BER must be between 0 and 1")
    return 1 - (1 - ber) ** 100  # Assuming packet size of 100 bits

def export_to_csv(matrix, filename):
    matrix.to_csv(filename)

def export_to_json(matrix, filename):
    matrix_dict = matrix.applymap(lambda x: x if isinstance(x, dict) else str(x)).to_dict()
    with open(filename, 'w') as f:
        json.dump(matrix_dict, f, indent=4)
        
def visualize_matrix(matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix.isnull(), cbar=False, cmap='viridis')
    plt.title('Quality Matrix Visualization')
    plt.show()