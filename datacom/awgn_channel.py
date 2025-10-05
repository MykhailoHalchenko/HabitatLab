"""AWGN channel simulation with realistic distance-SNR relationship."""
import numpy as np
import tensorflow as tf

def calculate_distance_to_snr(distance, reference_snr_db=50, environment="indoor"):
    """
    Calculate realistic SNR based on distance.
    
    Args:
        distance: Distance in meters
        reference_snr_db: SNR at 1m reference distance
        environment: "indoor" or "free_space"
    
    Returns:
        float: SNR in dB
    """
    if distance <= 0:
        return float('inf')
    
    if distance < 1:
        distance = 1
    
    # Realistic path loss models
    if environment == "indoor":
        # Indoor path loss (more aggressive)
        if distance <= 5:
            path_loss_db = 20 * np.log10(distance)  # Free space-like
        elif distance <= 15:
            path_loss_db = 20 * np.log10(5) + 35 * np.log10(distance/5)  # Obstacles
        elif distance <= 30:
            path_loss_db = 20 * np.log10(5) + 35 * np.log10(15/5) + 40 * np.log10(distance/15)
        else:
            path_loss_db = 20 * np.log10(5) + 35 * np.log10(15/5) + 40 * np.log10(30/15) + 45 * np.log10(distance/30)
    else:  # free_space
        # Free space path loss (less aggressive)
        path_loss_db = 20 * np.log10(distance) + 20 * np.log10(2.4) + 32.44  # Friis equation
    
    snr_db = reference_snr_db - path_loss_db
    
    # Realistic minimum SNR (thermal noise floor around -90 to -100 dBm)
    return max(float(snr_db), -20)

def transmit_data(bits, snr_db):
    """Transmit data through AWGN channel."""
    bits_tf = tf.cast(bits, tf.float32)
    symbols = 2 * bits_tf - 1  # BPSK modulation: 0->-1, 1->+1
    
    snr_db = tf.convert_to_tensor(snr_db, dtype=tf.float32)
    snr_linear = tf.pow(10.0, snr_db / 10.0)
    noise_variance = 1.0 / (2.0 * snr_linear)
    noise = tf.random.normal(tf.shape(symbols), stddev=tf.sqrt(noise_variance), dtype=tf.float32)
    
    received = symbols + noise
    return received

def calculate_ber(original_bits, received_symbols):
    """Calculate BER from received symbols."""
    received_bits = tf.cast(received_symbols > 0, tf.int32)
    original_bits_int = tf.cast(original_bits, tf.int32)
    
    errors = tf.reduce_sum(tf.cast(original_bits_int != received_bits, tf.float32))
    ber = errors / tf.cast(tf.size(original_bits), tf.float32)
    return float(ber.numpy())

def simulate_transmission(mod_a, mod_b, distance=None, snr_db=None, num_bits=100000, environment="indoor"):
    """Simulate transmission between two modules."""
    tf.random.set_seed(42)  # For reproducible results
    
    if snr_db is None and distance is not None:
        snr_db = calculate_distance_to_snr(distance, environment=environment)
    elif snr_db is None:
        raise ValueError("Either distance or snr_db must be provided")
    
    # Generate random bits
    bits = tf.random.uniform([num_bits], 0, 2, dtype=tf.int32)
    
    # Transmit through channel
    received = transmit_data(bits, snr_db)
    
    # Calculate BER
    ber = calculate_ber(bits, received)
    
    # Calculate capacity (theoretical maximum)
    capacity = calculate_capacity(snr_db)
    
    return {
        'ber': ber,
        'snr_db': float(snr_db),
        'capacity_bps_hz': capacity,
        'module_a': mod_a,
        'module_b': mod_b,
        'distance': distance if distance is not None else 0,
        'num_bits': num_bits,
        'environment': environment
    }

def calculate_capacity(snr_db):
    """Calculate Shannon channel capacity."""
    if snr_db <= -20:
        return 0.0
    
    snr_linear = 10 ** (snr_db / 10)
    capacity = np.log2(1 + snr_linear)
    return float(capacity)

def calculate_packet_error_rate(ber, packet_size_bits=1000):
    """Calculate Packet Error Rate from BER."""
    if ber <= 0:
        return 0.0
    if ber >= 1:
        return 1.0
    
    # More accurate PER calculation
    if ber < 1e-6:
        # Approximation for very low BER
        per = packet_size_bits * ber
    else:
        # Exact calculation
        per = 1 - (1 - ber) ** packet_size_bits
    
    return float(np.clip(per, 0.0, 1.0))

def test_channel_performance():
    """Test function to verify channel performance."""
    print("Testing AWGN channel performance:")
    print("-" * 40)
    
    # Test different distances
    distances = [1, 5, 10, 20, 50]
    print("\nDistance vs SNR (indoor environment):")
    for dist in distances:
        snr = calculate_distance_to_snr(dist, environment="indoor")
        print(f"  {dist:2d}m -> SNR: {snr:6.2f} dB")
    
    # Test different SNRs
    print("\nSNR vs BER performance:")
    snr_values = [-5, 0, 5, 10, 15, 20]
    for snr in snr_values:
        result = simulate_transmission("Test_A", "Test_B", snr_db=snr, num_bits=50000)
        per = calculate_packet_error_rate(result['ber'])
        print(f"  SNR: {snr:3d} dB -> BER: {result['ber']:8.6f}, PER: {per:6.4f}")


if __name__ == "__main__":
    print("Testing AWGN channel with realistic models...")
    
    # Test basic transmission
    result = simulate_transmission("ModuleA", "ModuleB", snr_db=10)
    print(f"\nTest 1: SNR=10dB")
    print(f"  BER: {result['ber']:.6f}")
    print(f"  Capacity: {result['capacity_bps_hz']:.2f} bps/Hz")
    
    # Test distance-based transmission
    result = simulate_transmission("ModuleA", "ModuleB", distance=10)
    print(f"\nTest 2: Distance=10m")
    print(f"  SNR: {result['snr_db']:.2f} dB")
    print(f"  BER: {result['ber']:.6f}")
    
    # Run performance tests
    test_channel_performance()
    
    print("\nAll tests completed!")