"""Quality matrix calculation for network links."""
import numpy as np
import pandas as pd
import json
import sys
import os

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datacom.network_topology import NetworkTopology
else:
    from .network_topology import NetworkTopology

def calculate_quality_matrix(network):
    """Calculate quality matrix for all module pairs."""
    module_ids = list(network.modules.keys())
    matrix = {}
    
    for mod_a in module_ids:
        matrix[mod_a] = {}
        for mod_b in module_ids:
            if mod_a == mod_b:
                matrix[mod_a][mod_b] = 0.0
            else:
                # Get SNR from graph edge data
                if network.graph.has_edge(mod_a, mod_b):
                    snr_db = network.graph[mod_a][mod_b].get('snr_db', 0)
                    # Convert SNR to realistic BER
                    ber = _snr_to_ber(snr_db)
                    per = calculate_per(ber)
                    matrix[mod_a][mod_b] = per
                else:
                    matrix[mod_a][mod_b] = 1.0  # No connection
    
    return matrix

def _snr_to_ber(snr_db):
    """Convert SNR to realistic BER values."""
    if snr_db > 20:
        return 1e-6
    elif snr_db > 15:
        return 1e-5
    elif snr_db > 10:
        return 1e-4
    elif snr_db > 5:
        return 1e-3
    elif snr_db > 0:
        return 1e-2
    elif snr_db > -5:
        return 0.1
    else:
        return 0.5

def calculate_per(ber, packet_size_bits=1000):
    """Calculate Packet Error Rate from BER."""
    if ber <= 0:
        return 0.0
    if ber >= 1:
        return 1.0
    
    # Use more realistic PER calculation
    if ber < 1e-4:
        # For very low BER, approximate as linear
        per = packet_size_bits * ber
    else:
        # Exact formula for higher BER
        per = 1 - (1 - ber) ** packet_size_bits
    
    return float(np.clip(per, 0.0, 1.0))

def export_to_csv(matrix, filename):
    """Export matrix to CSV."""
    df = pd.DataFrame(matrix)
    df.to_csv(filename)

def export_to_json(matrix, filename):
    """Export matrix to JSON."""
    with open(filename, 'w') as f:
        json.dump(matrix, f, indent=4)


if __name__ == "__main__":
    print("Testing quality matrix module...")
    from datacom.network_topology import NetworkTopology
    
    network = NetworkTopology()
    network.add_module("A", "power", (0, 0))
    network.add_module("B", "comms", (10, 0))
    network.create_graph()
    
    matrix = calculate_quality_matrix(network)
    print(f"\nQuality matrix: {matrix}")
    
    print("\nTesting PER calculation:")
    for ber in [0.000001, 0.0001, 0.001, 0.01, 0.1]:
        per = calculate_per(ber)
        print(f"  BER={ber:.6f} -> PER={per:.6f}")