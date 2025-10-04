import networkx as nx
import numpy as np
from network_topology import NetworkTopology

def simulate_failure(topology, failed_modules):
    for module in failed_modules:
        topology.remove_module(module)
    topology.create_graph()
    return topology

def assess_network_resilience(network, snr_db_values):
    resilience_report = {}
    MIN_SNR_THRESHOLD = 5  # dB
    
    for snr_db in snr_db_values:
        # Create temporary graph with only links above SNR threshold
        temp_graph = network.graph.copy()
        
        # Remove edges with insufficient SNR
        edges_to_remove = []
        for u, v, data in temp_graph.edges(data=True):
            if data.get('snr_db', float('inf')) < snr_db:
                edges_to_remove.append((u, v))
        
        temp_graph.remove_edges_from(edges_to_remove)
        
        resilience_report[snr_db] = {
            'is_connected': nx.is_connected(temp_graph),
            'num_nodes': temp_graph.number_of_nodes(),
            'num_edges': temp_graph.number_of_edges(),
            'removed_edges': len(edges_to_remove)
        }
    
    return resilience_report

def check_connectivity(network):
    return nx.is_connected(network.graph)

def find_disconnected_components(network, failed_module_id):
    temp_graph = network.graph.copy()
    if failed_module_id in temp_graph:
        temp_graph.remove_node(failed_module_id)
    return list(nx.connected_components(temp_graph))

def calculate_network_resilence(network):
    return nx.node_connectivity(network.graph)

def get_alternative_paths(network, source, target):
    try:
        paths = list(nx.all_simple_paths(network.graph, source=source, target=target))
        return paths
    except nx.NetworkXNoPath:
        return []