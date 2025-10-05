import networkx as nx
import copy
import sys
import os

# Fix imports for standalone execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datacom.network_topology import NetworkTopology
else:
    from .network_topology import NetworkTopology

def simulate_failure(topology, failed_modules):
    """Simulate module failures without modifying original."""
    temp_topology = copy.deepcopy(topology)
    for module in failed_modules:
        if module in temp_topology.modules:
            temp_topology.remove_module(module)
    temp_topology.create_graph()
    return temp_topology

def assess_network_resilience(network, snr_thresholds):
    """Assess resilience under different SNR conditions."""
    resilience_report = {}
    
    for threshold in snr_thresholds:
        temp_graph = network.graph.copy()
        edges_to_remove = [
            (u, v) for u, v, data in temp_graph.edges(data=True)
            if data.get('snr_db', float('inf')) < threshold
        ]
        temp_graph.remove_edges_from(edges_to_remove)
        
        resilience_report[threshold] = {
            'is_connected': nx.is_connected(temp_graph) if temp_graph.number_of_nodes() > 0 else True,
            'num_nodes': temp_graph.number_of_nodes(),
            'num_edges': temp_graph.number_of_edges(),
            'failed_links': len(edges_to_remove)
        }
    
    return resilience_report

def check_connectivity(network):
    """Check if network is fully connected."""
    return nx.is_connected(network.graph) if network.graph.number_of_nodes() > 0 else True

def find_disconnected_components(network, failed_module_id):
    """Find disconnected components after module failure."""
    temp_graph = network.graph.copy()
    if failed_module_id in temp_graph:
        temp_graph.remove_node(failed_module_id)
    return sorted(nx.connected_components(temp_graph), key=len, reverse=True)

def calculate_network_resilience(network):
    """Calculate node connectivity."""
    if network.graph.number_of_nodes() < 2:
        return 0
    return nx.node_connectivity(network.graph)

def get_alternative_paths(network, source, target, max_paths=5):
    """Find alternative paths between modules."""
    if source not in network.modules or target not in network.modules:
        return []
    if source == target:
        return [[source]]
    
    try:
        paths = list(nx.all_simple_paths(network.graph, source, target, cutoff=5))
        return paths[:max_paths]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


if __name__ == "__main__":
    print("Testing failure simulation...")
    from datacom.network_topology import NetworkTopology
    
    network = NetworkTopology()
    network.add_module("A", "power", (0, 0))
    network.add_module("B", "comms", (10, 0))
    network.add_module("C", "medical", (5, 10))
    network.create_graph()
    
    print(f"\nInitial connectivity: {check_connectivity(network)}")
    print(f"Resilience: {calculate_network_resilience(network)}")
    
    failed = simulate_failure(network, ["A"])
    print(f"\nAfter failure: {check_connectivity(failed)}")
    print(f"Remaining nodes: {failed.graph.number_of_nodes()}")