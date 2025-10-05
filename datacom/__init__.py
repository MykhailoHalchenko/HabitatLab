from . import sionna_patch  

from .awgn_channel import simulate_transmission, calculate_distance_to_snr
from .network_topology import NetworkTopology
from .quality_matrix import calculate_quality_matrix, calculate_per
from .failure_simulation import (
    simulate_failure, 
    check_connectivity,
    find_disconnected_components,
    calculate_network_resilience,
    get_alternative_paths,
    assess_network_resilience
)

# Спрощені імпорти для cellular_network
from .cellular_network import (
    Cellular5GSimulator,
    Cellular6GSimulator,
    compare_5g_6g
)

__version__ = "0.1.0"