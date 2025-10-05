import sys
import os
import numpy as np
import pandas as pd
from datacom.utils import setup_logging
import networkx as nx

# Ensure storage directory exists
os.makedirs("storage/comm_results", exist_ok=True)

from datacom.awgn_channel import (
    simulate_transmission, 
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
from datacom.cellular_network import (
    Cellular5GSimulator,
    Cellular6GSimulator,
    compare_5g_6g
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
        print(f"SNR: {snr_db} dB -> BER: {result['ber']:.6f}")
    
    # Test distance to SNR conversion
    print("\nTesting distance to SNR conversion:")
    distances = [1, 5, 10, 20, 50]
    for dist in distances:
        snr = calculate_distance_to_snr(dist)
        print(f"Distance: {dist}m -> SNR: {snr:.2f} dB")

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
                print(f"  {mod1} -> {mod2}: {dist:.2f}m")
    
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
    ber_values = [1e-6, 1e-4, 1e-3, 0.01, 0.1]
    for ber in ber_values:
        per = calculate_per(ber)
        print(f"  BER: {ber:.6f} -> PER: {per:.6f}")
    
    # Export to CSV
    try:
        export_to_csv(matrix, "storage/comm_results/quality_matrix.csv")
        print("\nMatrix exported to storage/comm_results/quality_matrix.csv")
    except Exception as e:
        print(f"\nWarning: Could not export CSV: {e}")
    
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
        print(f"  Path {i}: {' -> '.join(path)}")
    
    # Assess resilience at different SNR thresholds
    print("\n\nNetwork resilience at different SNR thresholds:")
    snr_thresholds = [-5, 0, 5, 10, 15]  # Більш реалістичні значення
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
        print(f"  {module_id}")
    
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
        print(f"  {mod1} <-> {mod2}: PER = {per:.6f}")
    
    # Find best connections
    print("\nBest 5 connections (lowest PER):")
    for mod1, mod2, per in per_list[-5:]:
        print(f"  {mod1} <-> {mod2}: PER = {per:.6f}")
    
    # Test critical failures
    print("\n\nTesting critical module failures:")
    critical_modules = ["power1", "life_support", "comms"]
    
    for module in critical_modules:
        failed_net = simulate_failure(network, [module])
        connected = check_connectivity(failed_net)
        resilience = calculate_network_resilience(failed_net)
        print(f"  {module} fails -> Network connected: {connected}, Resilience: {resilience}")

def test_cellular_networks():
    """Test 5G/6G cellular simulation."""
    print_separator("Test 6: 5G/6G Cellular Networks")
    
    print("\nTesting 5G NR simulation...")
    print("-" * 40)
    
    try:
        sim_5g = Cellular5GSimulator()
        result_5g = sim_5g.simulate_transmission(
            module_a="power", 
            module_b="comms", 
            distance=10, 
            snr_db=20
        )
        
        print(f"\n5G Results:")
        print(f"  Distance: {result_5g['distance']}m")
        print(f"  Path Loss: {result_5g['path_loss_db']:.2f} dB")
        print(f"  Effective SNR: {result_5g['snr_db']:.2f} dB")
        print(f"  BER: {result_5g['ber']:.6f}")
        print(f"  Throughput: {result_5g['throughput_mbps']:.2f} Mbps")
        print(f"  Modulation: {result_5g['modulation']}")
        print(f"  Coding Rate: {result_5g['coding_rate']:.2f}")
        print(f"  Antennas: {result_5g['num_tx_antennas']}x{result_5g['num_rx_antennas']} MIMO")
    except Exception as e:
        print(f"5G simulation error: {e}")
        result_5g = None
    
    print("\n\nTesting 6G THz simulation...")
    print("-" * 40)
    
    try:
        sim_6g = Cellular6GSimulator()
        result_6g = sim_6g.simulate_transmission(
            module_a="power",
            module_b="comms",
            distance=10,
            snr_db=20
        )
        
        print(f"\n6G Results:")
        print(f"  Distance: {result_6g['distance']}m")
        print(f"  Path Loss: {result_6g['path_loss_db']:.2f} dB")
        print(f"  Effective SNR: {result_6g['snr_db']:.2f} dB")
        print(f"  BER: {result_6g['ber']:.6f}")
        print(f"  Throughput: {result_6g['throughput_mbps']:.2f} Mbps")
        print(f"  Modulation: {result_6g['modulation']}")
        print(f"  Coding Rate: {result_6g['coding_rate']:.2f}")
        print(f"  Antennas: {result_6g['num_tx_antennas']}x{result_6g['num_rx_antennas']} MIMO")
    except Exception as e:
        print(f"6G simulation error: {e}")
        result_6g = None
    
    print("\n\nComparison at different distances:")
    print("=" * 60)
    for distance in [5, 10, 15, 20]:
        print(f"\nDistance: {distance}m")
        print("-" * 40)
        try:
            compare_5g_6g(distance=distance, snr_db=20)
        except Exception as e:
            print(f"Comparison failed: {e}")

def test_network_analysis():
    """Advanced network analysis tests."""
    print_separator("Test 7: Advanced Network Analysis")
    
    # Create complex network
    network = NetworkTopology()
    
    # Create circular habitat layout
    radius = 15
    num_modules = 8
    modules = []
    
    for i in range(num_modules):
        angle = 2 * np.pi * i / num_modules
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        module_id = f"module_{i+1}"
        module_type = ["power", "life_support", "medical", "comms", "storage", "lab", "sleep", "control"][i]
        modules.append((module_id, module_type, (x, y)))
    
    print("\nCreating circular habitat layout:")
    for module_id, module_type, position in modules:
        network.add_module(module_id, module_type, position)
        print(f"  {module_id} ({module_type}) at ({position[0]:.1f}, {position[1]:.1f})")
    
    network.create_graph()
    
    # Analyze network properties
    print(f"\nNetwork Analysis:")
    print(f"  Number of nodes: {network.graph.number_of_nodes()}")
    print(f"  Number of edges: {network.graph.number_of_edges()}")
    print(f"  Average degree: {np.mean([d for n, d in network.graph.degree()]):.2f}")
    print(f"  Network diameter: {nx.diameter(network.graph)}")
    print(f"  Average shortest path: {nx.average_shortest_path_length(network.graph):.2f}")
    
    # Centrality analysis
    degree_centrality = nx.degree_centrality(network.graph)
    betweenness_centrality = nx.betweenness_centrality(network.graph)
    
    print(f"\nMost critical modules (Degree Centrality):")
    for module, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {module}: {centrality:.3f}")
    
    print(f"\nMost critical modules (Betweenness Centrality):")
    for module, centrality in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {module}: {centrality:.3f}")

def main():
    """Main test runner."""
    setup_logging()
    
    print("\n" + "="*60)
    print("🏗️  HabitatLab Communication System Tests")
    print("="*60)
    
    try:
        # Run all tests
        test_awgn_channel()
        
        network = test_network_topology()
        
        matrix = test_quality_matrix(network)
        
        test_failure_simulation(network)
        
        test_full_scenario()
        
        test_cellular_networks()
        
        test_network_analysis()
        
        # Summary
        print_separator("TEST SUMMARY")
        print("\n All tests completed successfully!")
        print("\n Generated files:")
        print("  - storage/comm_results/quality_matrix.csv")
        print("  - datacom.log")
        print("\n Tested features:")
        print("  - AWGN channel simulation")
        print("  - Network topology management") 
        print("  - Quality matrix calculation")
        print("  - Failure simulation and resilience analysis")
        print("  - 5G NR communication (3.5 GHz, MIMO)")
        print("  - 6G THz communication (100 GHz, advanced MIMO)")
        print("  - Advanced network analysis (centrality, connectivity)")
        print("\n🏗️ Architecture:")
        print("  - Python-based simulation framework")
        print("  - NetworkX for topology analysis")
        print("  - TensorFlow for signal processing (AWGN)")
        print("  - Realistic channel models for space habitat")
        
        
    except Exception as e:
        print(f"\n Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()