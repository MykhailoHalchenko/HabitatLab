"""AWGN channel simulation without ray tracing dependencies."""
import numpy as np
import tensorflow as tf

def calculate_distance_to_snr(distance, reference_snr_db=40):
    """Calculate SNR based on distance using simplified model."""
    if distance <= 0:
        return float('inf')
    
    # More realistic distance-SNR relationship
    if distance <= 1:
        snr_db = reference_snr_db
    elif distance <= 5:
        snr_db = reference_snr_db - 10 * np.log10(distance)
    elif distance <= 20:
        snr_db = reference_snr_db - 15 * np.log10(distance)
    elif distance <= 50:
        snr_db = reference_snr_db - 20 * np.log10(distance)
    else:
        snr_db = reference_snr_db - 25 * np.log10(distance)
    
    return max(float(snr_db), -5)  # Realistic minimum

def transmit_data(bits, snr_db):
    """Transmit data through AWGN channel."""
    bits_tf = tf.cast(bits, tf.float32)
    symbols = 2 * bits_tf - 1
    
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

def simulate_transmission(mod_a, mod_b, distance=None, snr_db=None, num_bits=100000):
    """Simulate transmission between two modules."""
    tf.random.set_seed(42)
    
    if snr_db is None and distance is not None:
        snr_db = calculate_distance_to_snr(distance)
    elif snr_db is None:
        raise ValueError("Either distance or snr_db must be provided")
    
    bits = tf.random.uniform([num_bits], 0, 2, dtype=tf.int32)
    received = transmit_data(bits, snr_db)
    ber = calculate_ber(bits, received)
    
    return {
        'ber': ber,
        'snr_db': float(snr_db),
        'module_a': mod_a,
        'module_b': mod_b,
        'num_bits': num_bits
    }

def calculate_capacity(snr_db):
    """Calculate channel capacity (Shannon)."""
    snr_linear = 10 ** (snr_db / 10)
    capacity = np.log2(1 + snr_linear)
    return float(capacity)


if __name__ == "__main__":
    print("Testing AWGN channel...")
    
    result = simulate_transmission("ModuleA", "ModuleB", snr_db=10)
    print(f"\nTest 1: SNR=10dB")
    print(f"  BER: {result['ber']:.6f}")
    
    result = simulate_transmission("ModuleA", "ModuleB", distance=10)
    print(f"\nTest 2: Distance=10m")
    print(f"  SNR: {result['snr_db']:.2f} dB")
    print(f"  BER: {result['ber']:.6f}")
    
    print("\nAll tests completed!")