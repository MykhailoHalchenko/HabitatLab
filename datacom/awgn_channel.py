"""AWGN channel simulation without ray tracing dependencies."""
import numpy as np

# Import only needed components from Sionna (avoid rt module)
import tensorflow as tf
from sionna.channel import AWGN
from sionna.utils import compute_ber

def create_awgn_channel(snr_db):
    """Create AWGN channel."""
    return AWGN()

def transmit_data(bits, snr_db):
    """Transmit data through AWGN channel."""
    # BPSK Modulation: 0 -> -1, 1 -> +1
    bits_tf = tf.cast(bits, tf.float32)
    symbols = 2 * bits_tf - 1
    
    # Add noise with specified SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / snr_linear
    noise = tf.random.normal(tf.shape(symbols), stddev=tf.sqrt(noise_power))
    
    received = symbols + noise
    return received

def calculate_ber(original_bits, received_symbols):
    """Calculate BER from received symbols."""
    # Hard decision: positive -> 1, negative -> 0
    received_bits = tf.cast(received_symbols > 0, tf.int32)
    original_bits_int = tf.cast(original_bits, tf.int32)
    
    errors = tf.reduce_sum(tf.cast(original_bits_int != received_bits, tf.float32))
    ber = errors / tf.cast(tf.size(original_bits), tf.float32)
    return ber.numpy()

def simulate_transmission(mod_a, mod_b, distance=None, snr_db=None, num_bits=1000):
    """
    Simulate transmission between two modules.
    
    Args:
        mod_a: Source module
        mod_b: Destination module
        distance: Distance in meters (optional)
        snr_db: SNR in dB (optional, calculated from distance if not provided)
        num_bits: Number of bits to transmit
    
    Returns:
        dict: Transmission results including BER
    """
    # Calculate SNR from distance if not provided
    if snr_db is None and distance is not None:
        snr_db = calculate_distance_to_snr(distance)
    elif snr_db is None:
        raise ValueError("Either distance or snr_db must be provided")
    
    # Generate random bits
    bits = tf.random.uniform([num_bits], 0, 2, dtype=tf.int32)
    
    # Transmit through channel
    received = transmit_data(bits, snr_db)
    
    # Calculate BER
    ber = calculate_ber(bits, received)
    
    return {
        'ber': float(ber),
        'snr_db': float(snr_db),
        'module_a': mod_a,
        'module_b': mod_b,
        'num_bits': num_bits
    }

def calculate_distance_to_snr(distance, path_loss_exponent=3.5, 
                               reference_distance=1.0, reference_snr_db=30, 
                               frequency_ghz=2.4):
    """
    Calculate SNR based on distance using path loss model.
    
    Args:
        distance: Distance in meters
        path_loss_exponent: Path loss exponent (3.5 for indoor)
        reference_distance: Reference distance in meters
        reference_snr_db: SNR at reference distance
        frequency_ghz: Carrier frequency in GHz
    
    Returns:
        float: SNR in dB
    """
    if distance <= 0:
        raise ValueError("Distance must be positive")
    
    if distance < reference_distance:
        distance = reference_distance
    
    # Path loss model
    path_loss_db = 10 * path_loss_exponent * np.log10(distance / reference_distance)
    snr_db = reference_snr_db - path_loss_db
    
    return max(snr_db, -10)  # Minimum SNR threshold

def calculate_capacity(snr_db):
    """Calculate channel capacity (Shannon)."""
    snr_linear = 10 ** (snr_db / 10)
    capacity = 0.5 * np.log2(1 + snr_linear)
    return capacity

def calculate_ser(original_symbols, received_symbols):
    """Calculate Symbol Error Rate."""
    errors = tf.reduce_sum(tf.cast(original_symbols != received_symbols, tf.float32))
    ser = errors / tf.cast(tf.size(original_symbols), tf.float32)
    return ser.numpy()

def calculate_mse(original_symbols, received_symbols):
    """Calculate Mean Squared Error."""
    mse = tf.reduce_mean(tf.square(original_symbols - received_symbols))
    return mse.numpy()

def calculate_rmse(original_symbols, received_symbols):
    """Calculate Root Mean Squared Error."""
    mse = calculate_mse(original_symbols, received_symbols)
    return np.sqrt(mse)


# Test if run directly
if __name__ == "__main__":
    print("Testing AWGN channel...")
    
    # Test 1: Basic transmission
    result = simulate_transmission("ModuleA", "ModuleB", snr_db=10)
    print(f"\nTest 1: SNR=10dB")
    print(f"  BER: {result['ber']:.6f}")
    
    # Test 2: Distance-based
    result = simulate_transmission("ModuleA", "ModuleB", distance=10)
    print(f"\nTest 2: Distance=10m")
    print(f"  SNR: {result['snr_db']:.2f} dB")
    print(f"  BER: {result['ber']:.6f}")
    
    # Test 3: Different SNR values
    print("\nTest 3: BER vs SNR")
    for snr in [0, 5, 10, 15, 20]:
        result = simulate_transmission("A", "B", snr_db=snr)
        print(f"  SNR={snr}dB: BER={result['ber']:.6f}")
    
    print("\n✓ All tests completed!")