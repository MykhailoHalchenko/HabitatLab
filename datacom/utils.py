import numpy as np
import matplotlib.pyplot as plt
import logging
import networkx as nx
import json
from datetime import datetime

DEFAULT_SNR_DB = 10
SPEED_OF_LIGHT = 3e8
CARRIER_FREQUENCY = 2.4e9
MAX_DISTANCE = 100

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("datacom.log"),
                                  logging.StreamHandler()])
    logging.info("Logging is set up.")

def distance_to_snr(distance):
    if distance <= 0:
        return DEFAULT_SNR_DB
    path_loss = 20 * np.log10(distance) + 20 * np.log10(CARRIER_FREQUENCY) - 147.55
    snr_db = DEFAULT_SNR_DB - path_loss
    return max(snr_db, 0)

def ber_to_per(ber, packet_size=100):
    if ber < 0 or ber > 1:
        raise ValueError("BER must be between 0 and 1")
    return 1 - (1 - ber) ** packet_size

def save_results(data, filename=None):
    if filename is None:
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_results(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        
def plot_network_graph(network):
    pos = {node: (data['position'][0], data['position'][1]) for node, data in network.graph.nodes(data=True)}
    edge_labels = {(u, v): f"SNR: {data.get('snr_db', 'N/A'):.1f} dB" for u, v, data in network.graph.edges(data=True)}
    
    plt.figure(figsize=(10, 8))
    nx.draw(network.graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    nx.draw_networkx_edge_labels(network.graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Network Topology")
    plt.show()
    
def generate_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')