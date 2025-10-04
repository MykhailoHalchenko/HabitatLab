"""
Main testing script for HabitatLab communication system.
Tests all modules from datacom package.
"""

import sys
import numpy as np
import pandas as pd
from datacom.awgn_channel import (
    simulate_transmission, 
    calculate_ber,
    calculate_distance_to_snr
)
from datacom.network_topology import NetworkTopology
from datacom.quality_matrix import (
    calculate_quality_matrix,
    calculate_per,
    export_to_csv
)
from datacom.failure_simulation import (
    simulate_failure,
    check_connectivity,
    find_disconnected_components,
    calculate_network_resilience,
    get_alternative_paths,
    assess_network_resilience
)

def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "="*60)
    if title:
        print(f"  {title}")
        print("="*60)

def test_awgn_channel():
    """Test AWGN channel simulation."""
    print_separator("Test 1: AWGN Channel")
    
    # Test with different SNR values
    snr_values = [0, 5, 10, 15, 20]
    print("\nTesting transmission at different SNR levels:")
    
    for snr_db in snr_values:
        result = simulate_transmission("ModuleA", "ModuleB", snr_db=snr_db)
        print(f"SNR: {snr_db} dB → BER: {result['ber']:.6f}")
    
    # Test distance to SNR conversion
    print("\nTesting distance to SNR conversion:")
    distances = [1, 5, 10, 20, 50]
    for dist in distances:
        snr = calculate_distance_to_snr(dist)
        print(f"Distance: {dist}m → SNR: {snr:.2f} dB")

def test_network_topology():
    """Test network topology creation."""
    print_separator("Test 2: Network Topology")
    
    # Create network with 4 modules
    network = NetworkTopology()
    
    modules = [
        ("power", "power_unit", (0, 0)),
        ("life_support", "life_support", (10, 0)),
        ("medical", "medical", (0, 10)),
        ("comms", "communication", (10, 10))
    ]
    
    print("\nAdding modules:")
    for module_id, module_type, position in modules:
        network.add_module(module_id, module_type, position)
        print(f"  Added: {module_id} at {position}")
    
    # Create graph
    network.create_graph()
    print(f"\nNetwork created:")
    print(f"  Nodes: {network.graph.number_of_nodes()}")
    print(f"  Edges: {network.graph.number_of_edges()}")
    
    # Test distance calculation
    print("\nDistance matrix:")
    for mod1 in ["power", "life_support", "medical", "comms"]:
        for mod2 in ["power", "life_support", "medical", "comms"]:
            if mod1 != mod2:
                dist = network.calculate_distance(mod1, mod2)
                print(f"  {mod1} → {mod2}: {dist:.2f}m")
    
    return network

def test_quality_matrix(network):
    """Test quality matrix calculation."""
    print_separator("Test 3: Quality Matrix")
    
    # Calculate quality matrix
    print("\nCalculating quality matrix...")
    matrix = calculate_quality_matrix(network)
    
    print("\nQuality Matrix (Packet Error Rate):")
    df = pd.DataFrame(matrix)
    print(df.to_string())
    
    # Test PER calculation
    print("\n\nPER calculation for different BER values:")
    ber_values = [0.001, 0.01, 0.05, 0.1]
    for ber in ber_values:
        per = calculate_per(ber)
        print(f"  BER: {ber:.3f} → PER: {per:.6f}")
    
    # Export to CSV
    export_to_csv(matrix, "storage/comm_results/quality_matrix.csv")
    print("\n✓ Matrix exported to storage/comm_results/quality_matrix.csv")
    
    return matrix

def test_failure_simulation(network):
    """Test failure simulation."""
    print_separator("Test 4: Failure Simulation")
    
    # Check initial connectivity
    print("\nInitial network state:")
    print(f"  Connected: {check_connectivity(network)}")
    print(f"  Resilience: {calculate_network_resilience(network)}")
    
    # Simulate failure of one module
    print("\n\nSimulating failure of 'power' module:")
    failed_network = simulate_failure(network, ["power"])
    
    print(f"  Connected: {check_connectivity(failed_network)}")
    print(f"  Remaining nodes: {failed_network.graph.number_of_nodes()}")
    print(f"  Remaining edges: {failed_network.graph.number_of_edges()}")
    
    # Find disconnected components
    components = find_disconnected_components(network, "power")
    print(f"\n  Disconnected components: {len(components)}")
    for i, comp in enumerate(components, 1):
        print(f"    Component {i}: {comp}")
    
    # Test alternative paths
    print("\n\nAlternative paths from 'life_support' to 'medical':")
    paths = get_alternative_paths(network, "life_support", "medical")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: {' → '.join(path)}")
    
    # Assess resilience at different SNR thresholds
    print("\n\nNetwork resilience at different SNR thresholds:")
    snr_thresholds = [0, 5, 10, 15, 20]
    resilience = assess_network_resilience(network, snr_thresholds)
    
    for snr, data in resilience.items():
        print(f"\n  SNR threshold: {snr} dB")
        print(f"    Connected: {data['is_connected']}")
        print(f"    Active links: {data['num_edges']}")
        print(f"    Failed links: {data['failed_links']}")

def test_full_scenario():
    """Test complete scenario."""
    print_separator("Test 5: Complete Scenario")
    
    # Create larger network
    network = NetworkTopology()
    
    modules_layout = [
        ("power1", "power_unit", (0, 0)),
        ("power2", "power_unit", (20, 0)),
        ("life_support", "life_support", (10, 10)),
        ("medical", "medical", (5, 15)),
        ("sleep1", "sleep_quarters", (15, 15)),
        ("comms", "communication", (10, 20)),
        ("storage", "storage", (0, 20))
    ]
    
    print("\nCreating habitat with 7 modules:")
    for module_id, module_type, position in modules_layout:
        network.add_module(module_id, module_type, position)
        print(f"  ✓ {module_id}")
    
    network.create_graph()
    
    print(f"\nNetwork statistics:")
    print(f"  Total modules: {network.graph.number_of_nodes()}")
    print(f"  Total links: {network.graph.number_of_edges()}")
    print(f"  Connectivity: {calculate_network_resilience(network)}")
    
    # Calculate and display quality
    print("\nCalculating communication quality...")
    matrix = calculate_quality_matrix(network)
    
    # Find worst connections
    print("\nWorst 5 connections (highest PER):")
    per_list = []
    module_ids = list(network.modules.keys())
    for i, mod1 in enumerate(module_ids):
        for mod2 in module_ids[i+1:]:
            per_list.append((mod1, mod2, matrix[mod1][mod2]))
    
    per_list.sort(key=lambda x: x[2], reverse=True)
    for mod1, mod2, per in per_list[:5]:
        print(f"  {mod1} ↔ {mod2}: PER = {per:.6f}")
    
    # Test critical failures
    print("\n\nTesting critical module failures:")
    critical_modules = ["power1", "life_support", "comms"]
    
    for module in critical_modules:
        failed_net = simulate_failure(network, [module])
        connected = check_connectivity(failed_net)
        print(f"  {module} fails → Network connected: {connected}")

def main():
    """Main test runner."""
    print("\n" + "="*60)
    print("  HABITATLAB COMMUNICATION SYSTEM TEST SUITE")
    print("="*60)
    
    try:
        # Run all tests
        test_awgn_channel()
        
        network = test_network_topology()
        
        matrix = test_quality_matrix(network)
        
        test_failure_simulation(network)
        
        test_full_scenario()
        
        # Summary
        print_separator("TEST SUMMARY")
        print("\n✓ All tests completed successfully!")
        print("\nGenerated files:")
        print("  - storage/comm_results/quality_matrix.csv")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()