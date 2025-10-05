"""
5G/6G cellular network simulation for space habitat inter-module communication.
Realistic models with accurate path loss and throughput calculations.
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
        self.technology = "5G"
        
        # 5G parameters (will be adjusted based on SNR)
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
        
        # Calculate throughput (adaptive based on SNR)
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
            'technology': self.technology
        }
    
    def _calculate_path_loss(self, distance):
        """
        Calculate path loss using realistic 3GPP Indoor Hotspot (InH) model.
        """
        if distance < 1:
            distance = 1
        
        fc_ghz = self.carrier_frequency / 1e9
        
        # 3GPP InH path loss model with distance breakpoints
        if distance <= 10:
            # Short distance: PL = 32.4 + 17.3*log10(d) + 20*log10(fc)
            pl_db = 32.4 + 17.3 * np.log10(distance) + 20 * np.log10(fc_ghz)
        else:
            # Longer distance: higher path loss exponent
            pl_db = 32.4 + 17.3 * np.log10(10) + 20 * np.log10(fc_ghz) + \
                    10 * 2.5 * np.log10(distance/10)
        
        return min(pl_db, 120)  # Reasonable maximum
    
    def _calculate_ber(self, snr_db):
        """
        Calculate realistic BER based on SNR and modulation scheme.
        """
        if snr_db <= -5:
            return 0.5  # Random bits below -5 dB
        
        snr_linear = 10 ** (snr_db / 10)
        
        # Realistic BER approximation for 64-QAM in AWGN
        # Based on Q-function approximation for M-QAM
        if self.modulation == "64-QAM":
            ber = 0.2 * np.exp(-snr_linear / 21)
        elif self.modulation == "16-QAM":
            ber = 0.2 * np.exp(-snr_linear / 10)
        elif self.modulation == "QPSK":
            ber = 0.5 * np.exp(-snr_linear / 2)
        else:  # BPSK
            ber = 0.5 * np.exp(-snr_linear)
            
        # MIMO diversity gain
        diversity_gain = np.sqrt(self.num_tx_ant * self.num_rx_ant)
        effective_ber = ber / diversity_gain
        
        return float(np.clip(effective_ber, 1e-9, 0.5))
    
    def _calculate_throughput(self, snr_db):
        """
        Calculate realistic throughput with adaptive modulation and coding.
        """
        if snr_db <= 0:
            return 0.0
        
        # Adaptive Modulation and Coding (AMC) based on SNR
        if snr_db > 25:
            efficiency = 6.0  # 256-QAM
            coding_rate = 0.9
            self.modulation = "256-QAM"
            self.num_bits_per_symbol = 8
        elif snr_db > 20:
            efficiency = 4.5  # 64-QAM  
            coding_rate = 0.75
            self.modulation = "64-QAM"
            self.num_bits_per_symbol = 6
        elif snr_db > 15:
            efficiency = 3.0  # 16-QAM
            coding_rate = 0.6
            self.modulation = "16-QAM"
            self.num_bits_per_symbol = 4
        elif snr_db > 10:
            efficiency = 2.0  # QPSK
            coding_rate = 0.5
            self.modulation = "QPSK"
            self.num_bits_per_symbol = 2
        elif snr_db > 5:
            efficiency = 1.0  # BPSK
            coding_rate = 0.3
            self.modulation = "BPSK"
            self.num_bits_per_symbol = 1
        else:
            efficiency = 0.5  # Very low rate
            coding_rate = 0.2
            self.modulation = "BPSK"
            self.num_bits_per_symbol = 1
        
        self.coding_rate = coding_rate
        
        # MIMO spatial multiplexing
        spatial_streams = min(self.num_tx_ant, self.num_rx_ant)
        total_efficiency = efficiency * spatial_streams * coding_rate
        
        # Throughput in Mbps
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
        
        self.technology = "6G"
        # 6G enhanced parameters
        self.bandwidth = 400e6  # 400 MHz
        self.is_thz_band = carrier_frequency > 90e9
        
    def _calculate_path_loss(self, distance):
        """
        Calculate realistic path loss for THz band (6G model).
        """
        if distance < 0.1:
            distance = 0.1
        
        if self.is_thz_band:
            # THz path loss model with atmospheric absorption
            fc_thz = self.carrier_frequency / 1e12
            
            # Free space path loss
            pl_db = 20 * np.log10(distance) + 20 * np.log10(fc_thz) + 32.45
            
            # Atmospheric absorption (significant for THz)
            # Water vapor and oxygen absorption increase with frequency
            absorption_coeff = 0.3 * fc_thz  # dB/m (increases with frequency)
            pl_db += absorption_coeff * distance
            
            # Additional losses for indoor environment
            pl_db += 5  # Wall/obstacle penetration losses
            
        else:
            # Use standard 5G model for sub-THz
            pl_db = super()._calculate_path_loss(distance)
        
        return min(pl_db, 150)  # THz has higher maximum losses
    
    def _calculate_ber(self, snr_db):
        """
        Calculate BER for 6G with advanced coding and massive MIMO.
        """
        if snr_db <= -10:
            return 0.5  # THz more sensitive to low SNR
        
        snr_linear = 10 ** (snr_db / 10)
        
        # 6G uses advanced coding (LDPC, polar codes)
        if self.modulation == "256-QAM":
            ber = 0.15 * np.exp(-snr_linear / 42)
        elif self.modulation == "64-QAM":
            ber = 0.15 * np.exp(-snr_linear / 25)
        elif self.modulation == "16-QAM":
            ber = 0.2 * np.exp(-snr_linear / 12)
        elif self.modulation == "QPSK":
            ber = 0.3 * np.exp(-snr_linear / 3)
        else:  # BPSK
            ber = 0.4 * np.exp(-snr_linear)
            
        # Enhanced MIMO gains for 6G (massive MIMO)
        diversity_gain = np.sqrt(self.num_tx_ant * self.num_rx_ant) * 1.2
        effective_ber = ber / diversity_gain
        
        return float(np.clip(effective_ber, 1e-10, 0.5))
    
    def _calculate_throughput(self, snr_db):
        """
        Calculate 6G throughput with enhanced efficiency and massive MIMO.
        """
        if snr_db <= -5:
            return 0.0
        
        # 6G supports higher order modulation and better coding
        if snr_db > 30:
            efficiency = 8.0  # 1024-QAM possible in 6G
            coding_rate = 0.95
            self.modulation = "1024-QAM"
            self.num_bits_per_symbol = 10
        elif snr_db > 25:
            efficiency = 7.0  # 256-QAM
            coding_rate = 0.9
            self.modulation = "256-QAM"
            self.num_bits_per_symbol = 8
        elif snr_db > 20:
            efficiency = 6.0  # 64-QAM
            coding_rate = 0.85
            self.modulation = "64-QAM"
            self.num_bits_per_symbol = 6
        elif snr_db > 15:
            efficiency = 4.0  # 16-QAM
            coding_rate = 0.7
            self.modulation = "16-QAM"
            self.num_bits_per_symbol = 4
        elif snr_db > 10:
            efficiency = 2.5  # QPSK
            coding_rate = 0.6
            self.modulation = "QPSK"
            self.num_bits_per_symbol = 2
        elif snr_db > 5:
            efficiency = 1.2  # BPSK with better coding
            coding_rate = 0.4
            self.modulation = "BPSK"
            self.num_bits_per_symbol = 1
        else:
            efficiency = 0.8
            coding_rate = 0.3
            self.modulation = "BPSK"
            self.num_bits_per_symbol = 1
        
        self.coding_rate = coding_rate
        
        # Massive MIMO spatial multiplexing (more streams possible)
        spatial_streams = min(self.num_tx_ant, self.num_rx_ant) * 1.5  # 6G advantage
        total_efficiency = efficiency * spatial_streams * coding_rate
        
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
    print(f"  Coding Rate: {results_5g['coding_rate']:.2f}")
    print(f"  Antennas: {results_5g['num_tx_antennas']}x{results_5g['num_rx_antennas']} MIMO")
    
    print("\n6G Results (100 GHz THz):")
    print(f"  Path Loss: {results_6g['path_loss_db']:.2f} dB")
    print(f"  Effective SNR: {results_6g['snr_db']:.2f} dB")
    print(f"  BER: {results_6g['ber']:.6f}")
    print(f"  Throughput: {results_6g['throughput_mbps']:.2f} Mbps")
    print(f"  Modulation: {results_6g['modulation']}")
    print(f"  Coding Rate: {results_6g['coding_rate']:.2f}")
    print(f"  Antennas: {results_6g['num_tx_antennas']}x{results_6g['num_rx_antennas']} MIMO")
    
    # Performance comparison
    throughput_ratio = results_6g['throughput_mbps'] / max(results_5g['throughput_mbps'], 0.1)
    ber_ratio = results_6g['ber'] / max(results_5g['ber'], 1e-9)
    
    print(f"\nPerformance Comparison:")
    print(f"  Throughput (6G/5G): {throughput_ratio:.2f}x")
    print(f"  BER (6G/5G): {ber_ratio:.2f}x")
    
    if throughput_ratio > 1 and ber_ratio < 1:
        print("  → 6G performs better")
    elif throughput_ratio < 1 and ber_ratio > 1:
        print("  → 5G performs better")  
    else:
        print("  → Mixed performance")
    
    return {'5g': results_5g, '6g': results_6g}


if __name__ == "__main__":
    print("="*60)
    print("  Testing 5G/6G Cellular Network Simulation")
    print("="*60)
    
    print("\nTest 1: 5G transmission at different distances")
    print("-" * 50)
    sim_5g = Cellular5GSimulator()
    for dist in [5, 10, 20]:
        result = sim_5g.simulate_transmission("power", "comms", distance=dist, snr_db=20)
        print(f"Distance: {dist:2d}m -> SNR: {result['snr_db']:6.2f} dB, "
              f"BER: {result['ber']:.2e}, Throughput: {result['throughput_mbps']:6.1f} Mbps")
    
    print("\n\nTest 2: 5G vs 6G comparison")
    print("=" * 60)
    for dist in [5, 10, 15, 20]:
        compare_5g_6g(dist, snr_db=20)
    
    print("\n" + "="*60)
    print("  All tests completed!")
    print("="*60)