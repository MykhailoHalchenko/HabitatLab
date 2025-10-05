"""
5G/6G cellular network simulation for space habitat inter-module communication.
Simplified version without complex Sionna dependencies.
"""
import numpy as np
import tensorflow as tf
import sys
import os

class Cellular5GSimulator:
    """Simulate 5G NR-like communication between habitat modules."""
    
    def __init__(self, num_ofdm_symbols=14, fft_size=128, subcarrier_spacing=30e3,
                 carrier_frequency=3.5e9, num_tx_ant=2, num_rx_ant=2):
        """
        Initialize 5G simulator.
        """
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.subcarrier_spacing = subcarrier_spacing
        self.carrier_frequency = carrier_frequency
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        
        # 5G parameters
        self.modulation = "64-QAM"
        self.coding_rate = 0.7
        self.bandwidth = 100e6  # 100 MHz
        self.num_bits_per_symbol = 6  # for 64-QAM
        
    def simulate_transmission(self, module_a, module_b, distance, snr_db=20, num_frames=5):
        """
        Simulate 5G transmission between modules.
        """
        # Calculate path loss
        path_loss_db = self._calculate_path_loss(distance)
        effective_snr_db = snr_db - path_loss_db
        
        # Calculate BER based on SNR and modulation
        ber = self._calculate_ber(effective_snr_db)
        
        # Calculate throughput
        throughput_mbps = self._calculate_throughput(effective_snr_db)
        
        return {
            'ber': ber,
            'snr_db': effective_snr_db,
            'path_loss_db': path_loss_db,
            'throughput_mbps': throughput_mbps,
            'module_a': module_a,
            'module_b': module_b,
            'distance': distance,
            'modulation': self.modulation,
            'coding_rate': self.coding_rate,
            'num_tx_antennas': self.num_tx_ant,
            'num_rx_antennas': self.num_rx_ant,
            'technology': '5G'
        }
    
    def _calculate_path_loss(self, distance):
        """
        Calculate path loss using 3GPP Indoor Hotspot (InH) model.
        """
        if distance < 1:
            distance = 1
        
        # 3GPP InH path loss model: PL = 32.4 + 17.3*log10(d) + 20*log10(fc)
        fc_ghz = self.carrier_frequency / 1e9
        pl_db = 32.4 + 17.3 * np.log10(distance) + 20 * np.log10(fc_ghz)
        
        return pl_db
    
    def _calculate_ber(self, snr_db):
        """
        Calculate BER based on SNR and modulation scheme.
        """
        snr_linear = 10 ** (snr_db / 10)
        
        # Approximate BER for 64-QAM in AWGN
        if snr_db > 20:
            ber = 1e-6
        elif snr_db > 15:
            ber = 1e-4
        elif snr_db > 10:
            ber = 1e-3
        elif snr_db > 5:
            ber = 1e-2
        elif snr_db > 0:
            ber = 0.1
        else:
            ber = 0.5
            
        # Add MIMO diversity gain
        diversity_gain = np.log2(self.num_tx_ant * self.num_rx_ant)
        effective_ber = ber / diversity_gain
        
        return float(np.clip(effective_ber, 1e-7, 0.5))
    
    def _calculate_throughput(self, snr_db):
        """
        Calculate throughput using Shannon capacity with practical efficiency.
        """
        snr_linear = 10 ** (snr_db / 10)
        
        # Spectral efficiency (bits/s/Hz)
        if snr_db > 25:
            efficiency = 4.5  # 64-QAM high coding rate
        elif snr_db > 20:
            efficiency = 4.0
        elif snr_db > 15:
            efficiency = 3.0
        elif snr_db > 10:
            efficiency = 2.0
        elif snr_db > 5:
            efficiency = 1.0
        else:
            efficiency = 0.5
            
        # MIMO spatial multiplexing gain
        spatial_streams = min(self.num_tx_ant, self.num_rx_ant)
        total_efficiency = efficiency * spatial_streams * self.coding_rate
        
        throughput_mbps = (self.bandwidth * total_efficiency) / 1e6
        
        return throughput_mbps


class Cellular6GSimulator(Cellular5GSimulator):
    """Simulate advanced 6G features with THz bands and massive MIMO."""
    
    def __init__(self, carrier_frequency=100e9, num_tx_ant=4, num_rx_ant=4, **kwargs):
        """
        Initialize 6G simulator with THz carrier and enhanced MIMO.
        """
        super().__init__(
            carrier_frequency=carrier_frequency,
            num_tx_ant=num_tx_ant,
            num_rx_ant=num_rx_ant,
            **kwargs
        )
        
        # 6G enhanced parameters
        self.modulation = "256-QAM"
        self.coding_rate = 0.85
        self.bandwidth = 400e6  # 400 MHz
        self.num_bits_per_symbol = 8  # for 256-QAM
        self.is_thz_band = carrier_frequency > 90e9
        
    def _calculate_path_loss(self, distance):
        """
        Calculate path loss for THz band (6G model).
        """
        if distance < 0.1:
            distance = 0.1
        
        if self.is_thz_band:
            # THz path loss model
            fc_thz = self.carrier_frequency / 1e12
            
            # Free space path loss
            pl_db = 20 * np.log10(distance) + 20 * np.log10(fc_thz) + 32.45
            
            # Add atmospheric absorption (simplified model)
            absorption_db_per_m = 0.1 * fc_thz
            pl_db += absorption_db_per_m * distance
            
            return pl_db
        else:
            # Use standard 5G model for sub-THz
            return super()._calculate_path_loss(distance)
    
    def _calculate_ber(self, snr_db):
        """
        Calculate BER for 6G with advanced coding.
        """
        # 6G has better error correction
        snr_linear = 10 ** (snr_db / 10)
        
        if snr_db > 25:
            ber = 1e-7
        elif snr_db > 20:
            ber = 1e-5
        elif snr_db > 15:
            ber = 1e-4
        elif snr_db > 10:
            ber = 1e-3
        elif snr_db > 5:
            ber = 1e-2
        else:
            ber = 0.1
            
        # Enhanced MIMO gains for 6G
        diversity_gain = np.log2(self.num_tx_ant * self.num_rx_ant) * 1.5
        effective_ber = ber / diversity_gain
        
        return float(np.clip(effective_ber, 1e-8, 0.5))
    
    def _calculate_throughput(self, snr_db):
        """
        Calculate 6G throughput with enhanced efficiency.
        """
        snr_linear = 10 ** (snr_db / 10)
        
        # 6G spectral efficiency
        if snr_db > 30:
            efficiency = 8.0  # 256-QAM very high coding rate
        elif snr_db > 25:
            efficiency = 7.0
        elif snr_db > 20:
            efficiency = 6.0
        elif snr_db > 15:
            efficiency = 5.0
        elif snr_db > 10:
            efficiency = 4.0
        else:
            efficiency = 2.0
            
        # Massive MIMO spatial multiplexing
        spatial_streams = min(self.num_tx_ant, self.num_rx_ant)
        total_efficiency = efficiency * spatial_streams * self.coding_rate
        
        throughput_mbps = (self.bandwidth * total_efficiency) / 1e6
        
        return throughput_mbps


def compare_5g_6g(distance=10, snr_db=20):
    """
    Compare 5G and 6G performance at given distance.
    """
    print(f"\nComparing 5G vs 6G at distance = {distance}m, SNR = {snr_db} dB")
    
    # Simulate 5G
    sim_5g = Cellular5GSimulator()
    results_5g = sim_5g.simulate_transmission("Module_A", "Module_B", distance, snr_db)
    
    # Simulate 6G
    sim_6g = Cellular6GSimulator()
    results_6g = sim_6g.simulate_transmission("Module_A", "Module_B", distance, snr_db)
    
    print("\n5G Results (3.5 GHz):")
    print(f"  Path Loss: {results_5g['path_loss_db']:.2f} dB")
    print(f"  Effective SNR: {results_5g['snr_db']:.2f} dB")
    print(f"  BER: {results_5g['ber']:.6f}")
    print(f"  Throughput: {results_5g['throughput_mbps']:.2f} Mbps")
    print(f"  Modulation: {results_5g['modulation']}")
    print(f"  Antennas: {results_5g['num_tx_antennas']}x{results_5g['num_rx_antennas']} MIMO")
    
    print("\n6G Results (100 GHz THz):")
    print(f"  Path Loss: {results_6g['path_loss_db']:.2f} dB")
    print(f"  Effective SNR: {results_6g['snr_db']:.2f} dB")
    print(f"  BER: {results_6g['ber']:.6f}")
    print(f"  Throughput: {results_6g['throughput_mbps']:.2f} Mbps")
    print(f"  Modulation: {results_6g['modulation']}")
    print(f"  Antennas: {results_6g['num_tx_antennas']}x{results_6g['num_rx_antennas']} MIMO")
    
    return {'5g': results_5g, '6g': results_6g}


if __name__ == "__main__":
    print("="*60)
    print("  Testing 5G/6G Cellular Network Simulation")
    print("="*60)
    
    print("\nTest 1: 5G transmission")
    print("-" * 40)
    sim_5g = Cellular5GSimulator()
    result = sim_5g.simulate_transmission("power", "comms", distance=10, snr_db=20)
    print(f"Distance: {result['distance']}m")
    print(f"BER: {result['ber']:.6f}")
    print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")
    
    print("\n\nTest 2: 5G vs 6G comparison at different distances")
    print("=" * 60)
    for dist in [5, 10, 15]:
        compare_5g_6g(dist, snr_db=20)
    
    print("\n" + "="*60)
    print("  All tests completed!")
    print("="*60)