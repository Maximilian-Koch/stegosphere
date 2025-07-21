"""
A module for dividing payloads across several containers
"""
from numpy import random
import numpy as np

from stegosphere import selector, io

__all__ = ['roundrobin_chunks', 'split_encode','split_decode', 'weighted_chunks']

def weighted_chunks(payload, num_instances, weights, tol=1e-6):
    """Split a payload into chunks sized according to given weights.

    Args:
        payload: The full data string to split.
        num_instances: Number of desired chunks.
        weights: Sequence of floats summing to ~1. Length must equal num_instances.
        tol: Tolerance for sum(weights) == 1.0 check.

    Returns:
        A list of `num_instances` substrings in order.

    Raises:
        ValueError: If `len(weights) != num_instances` or weights don’t sum to ~1.
        TypeError: If payload is not a str.
    """
    if not isinstance(payload, str):
        raise TypeError(f'payload must be str, got {type(payload).__name__}')
    if len(weights) != num_instances:
        raise ValueError(f'Expected {num_instances} weights, got {len(weights)}')
    total = sum(weights)
    if abs(total - 1.0) > tol:
        raise ValueError(f'Weights must sum to 1.0±{tol}, but sum to {total}')

    length = len(payload)
    chunk_sizes = [int(round(w * length)) for w in weights]
    
    #fix rounding at last chunk
    diff = length - sum(chunk_sizes)
    chunk_sizes[-1] += diff
    
    chunks = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(payload[start:end])
        start = end
        
    return chunks


def roundrobin_chunks(payload, num_instances):
    """Distribute the payload characters in round‑robin order.

    Args:
        payload: The full data string to split.
        num_instances: How many interleaved sequences to produce.

    Returns:
        A list of `num_instances` strings, each containing every nth character.
    """
    rosters = [[] for _ in range(num_instances)]
    for i, bit in enumerate(payload):
        rosters[i % num_instances].append(bit)

    return [''.join(chunk) for chunk in rosters]

def reverse_roundrobin(payload, num_instances):
    """Reconstruct the original payload from round‑robin chunks.

    Args:
        payload: Concatenation of round‑robin‑distributed segments.
        num_instances: Number of segments used originally.

    Returns:
        The reassembled payload.
    """
    chunk_lengths = [len(payload) // num_instances] * num_instances
    remainder = len(payload) % num_instances
    
    for i in range(remainder):
        chunk_lengths[i] += 1
    
    chunks = []
    start = 0
    for length in chunk_lengths:
        chunks.append(payload[start:start + length])
        start += length
    

    output = [''] * len(payload)
    positions = [0] * num_instances
    for i in range(len(payload)):
        chunk_index = i % num_instances
        output[i] = chunks[chunk_index][positions[chunk_index]]
        positions[chunk_index] += 1
    
    return ''.join(output)


def split_encode(payload, instances, seed=None, distribution='even', distribution_args=None):
    """Encode a payload across multiple stego‐instances.

    Args:
        payload: The string to encode and split.
        instances: Callables (e.g. functions) that take a chunk and return stego‐data.
        seed: Optional PRNG seed to shuffle before splitting.
        distribution: One of {'even', 'weighted', 'roundrobin'}.
        distribution_args: Extra args for distribution methods
            (e.g. {'weights': [0.2, 0.8]}).

    Returns:
        A list of stego‐encoded outputs, one per instance.

    Raises:
        ValueError: On unknown distribution or bad args.
    """
    if not isinstance(payload, str):
        raise TypeError(f'payload must be str, got {type(payload).__name__}')
    instances = list(instances)
    n = len(instances)
    if n == 0:
        raise ValueError('At least one instance is required')
    if seed is not None:
        indices = selector.prng_indices(len(payload), seed)
        payload = ''.join(payload[i] for i in indices)
    else:
        indices = np.arange(len(payload))

    num_instances = len(instances)
    payload_chunks = []
    
    if distribution == 'even':
        chunk_length = len(payload)//num_instances
        for i in range(len(instances)):
            payload_chunks.append(payload[i*chunk_length:(i+1)*chunk_length])
        if remainder := len(payload) % len(instances):
            payload_chunks[-1] += payload[-remainder:]
    elif distribution == 'weighted':
        weights = distribution_args.get('weights')
        if weights is None:
            raise ValueError('Missing "weights" for weighted distribution')
        payload_chunks = weighted_chunks(payload, num_instances, weights)
    elif distribution == 'roundrobin':
        payload_chunks = roundrobin_chunks(payload, num_instances)
    else:
        raise ValueError(f'Unknown distribution "{distribution}"')

    outputs = []
    for inst, chunk in zip(instances, payload_chunks):
        try:
            outputs.append(inst(chunk))
        except Exception as e:
            raise RuntimeError(f'Instance {inst!r} failed to encode chunk: {e}') from e

    return outputs

    
def split_decode(instances, seed=None, distribution=None, distribution_args=None):
    """Decode payload from multiple stego‐instances.

    Args:
        instances: Callables returning stego‐data strings/bytes.
        seed: If provided, re‐shuffle back into original order.
        distribution: Same distribution used in encoding.
        distribution_args: Args for reverse operations (if needed).

    Returns:
        The reconstructed payload string.
    """
    if not instances:
        raise ValueError('At least one instance is required')
    output = ''
    for instance in instances:
        output += instance()
    if distribution == 'roundrobin':
        output = reverse_roundrobin(output, len(instances))
        
    if seed is None:
        return output
    else:
        indices = selector.prng_indices(len(output),seed)
        zeros = np.zeros(len(output),dtype=int)
        for bit, index in zip(output, indices):
            zeros[index] = bit
        return ''.join(map(str, zeros))
