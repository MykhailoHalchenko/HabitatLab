
import networkx as nx
import numpy as np
import sys
import os

# Fix imports for standalone execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from datacom.awgn_channel import simulate_transmission, calculate_distance_to_snr
else:
    from .awgn_channel import simulate_transmission, calculate_distance_to_snr

class NetworkTopology:
    def __init__(self):
        self.graph = nx.Graph()
        self.modules = {}
        
    def add_module(self, module_id, module_type, position):
        position = np.array(position, dtype=float)
        module_data = {'id': module_id, 'type': module_type, 'position': position}
        self.modules[module_id] = module_data
        self.graph.add_node(module_id, **module_data)
        
    def remove_module(self, module_id):
        if module_id in self.modules:
            del self.modules[module_id]
            if self.graph.has_node(module_id):
                self.graph.remove_node(module_id)
    
    def calculate_distance(self, module_a, module_b):
        """Calculate distance between two modules."""
        pos_a = self.modules[module_a]['position']
        pos_b = self.modules[module_b]['position']
        return float(np.linalg.norm(pos_a - pos_b))
    
    def create_graph(self):
        """Create fully connected graph with distances."""
        module_ids = list(self.modules.keys())
        for i, mod_a in enumerate(module_ids):
            for mod_b in module_ids[i+1:]:
                distance = self.calculate_distance(mod_a, mod_b)
                snr_db = calculate_distance_to_snr(distance)
                self.graph.add_edge(mod_a, mod_b, distance=distance, snr_db=snr_db)
                    
    def get_connections(self):
        return list(self.graph.edges(data=True))
    
    def calculate_link_quality(self, mod_a, mod_b):
        """Calculate link quality between two modules."""
        distance = self.calculate_distance(mod_a, mod_b)
        results = simulate_transmission(mod_a, mod_b, distance=distance)
        return results


if __name__ == "__main__":
    print("Testing network topology...")
    
    network = NetworkTopology()
    network.add_module("power", "power_unit", (0, 0))
    network.add_module("comms", "communication", (10, 0))
    network.add_module("medical", "medical", (5, 10))
    
    network.create_graph()
    
    print(f"\nNodes: {network.graph.number_of_nodes()}")
    print(f"Edges: {network.graph.number_of_edges()}")
    
    print("\nDistances:")
    for mod_a in network.modules:
        for mod_b in network.modules:
            if mod_a < mod_b:
                dist = network.calculate_distance(mod_a, mod_b)
                print(f"  {mod_a} -> {mod_b}: {dist:.2f}m")