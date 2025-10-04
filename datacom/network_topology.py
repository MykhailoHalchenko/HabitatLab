import networkx as nx
import numpy as np
from awgn_channel import *

class NetworkTopology:
    def __init__(modules, self):
        self.graph = nx.Graph()
        self.modules = modules  # List of module names
        
    
    def add_module(self, module_id, module_type, position):
        modules = {'id': module_id, 'type': module_type, 'position': position}
        self.modules[module_id] = modules
        self.graph.add_node(module_id, **modules)
        
    def remove_module(self, module_id):
        if module_id in self.modules:
            del self.modules[module_id]
            self.graph.remove_node(module_id)
    
    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def create_graph(self):
        for i, mod_a in enumerate(self.modules):
            for j, mod_b in enumerate(self.modules):
                if i < j:
                    pos_a = self.modules[mod_a]['position']
                    pos_b = self.modules[mod_b]['position']
                    distance = self.calculate_distance(pos_a, pos_b)
                    self.graph.add_edge(mod_a, mod_b, weight=distance)
                    
    def get_connections(self):
        return list(self.graph.edges(data=True))
    
    def calculate_link_quality(self, mod_a, mod_b, snr_db):
        distance = self.calculate_distance(self.modules[mod_a]['position'], self.modules[mod_b]['position'])
        snr_db = calculate_distance_to_snr(distance)
        results = simulate_transmission(mod_a, mod_b, snr_db)
        return results