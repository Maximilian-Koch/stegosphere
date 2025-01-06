from numpy import random
import numpy as np

from stegosphere import utils, io

__all__ = ['roundrobin_chunks', 'split_encode','split_decode', 'weighted_chunks']

def weighted_chunks(payload, num_instances, weights):
    """
    Splits payload into num_instances chunks according to weights.
    """
    assert sum(weights)==1
    assert len(weights)==num_instances

    length = len(payload)
    chunk_sizes = [int(round(w * length)) for w in weights]
    
    #fix rounding at last chunk
    diff = length - sum(chunk_sizes)
    if diff > 0:
        chunk_sizes[-1] += diff
    elif diff < 0:
        chunk_sizes[-1] += diff
    
    chunks = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(payload[start:end])
        start = end
        
    return chunks


def roundrobin_chunks(payload, num_instances):
    """
    Distribute payload in a cycle / round-robin.
    """
    rosters = [[] for _ in range(num_instances)]
    for i, bit in enumerate(payload):
        rosters[i % num_instances].append(bit)

    return [''.join(chunk) for chunk in rosters]

def reverse_roundrobin(payload, num_instances):
    """
    Reconstruct payload from round-robin payload.
    
    :param payload: The entire string distributed using round-robin.
    :type payload: str
    :param num_instances: The number of instances used
    :type num_instances: int
    
    :return: Reconstructed payload.
    :rtype: str
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
    """
    Encodes a payload across several instances.

    :param payload: The payload to be encoded.
    :type payload: str
    :param instances: The iterable of instances.
    :type instances: list
    :param seed: Seed to pseudo-randomly distribute the payload over the different instances.
    :type seed: int, optional
    :param distribution: How to distribute the payload across the instances. Defaults to even distribution.
    :type distribution: str, optional
    :param distribution_args: Additional distribution args
    :type distribution_args: dict, optional

    :return: outputs
    :rtype: list
    """

    
    if seed:
        indices = utils.prng_indices(len(payload), seed)
        payload = ''.join(payload[i] for i in indices)
    else:
        indices = np.arange(len(payload))

    num_instances = len(instances)
    payload_chunks = []
    output = []
    
    if distribution == 'even':
        chunk_length = len(payload)//num_instances
        for i in range(len(instances)):
            payload_chunks.append(payload[i*chunk_length:(i+1)*chunk_length])
        if remainder := len(payload) % len(instances):
            payload_chunks[-1] += payload[-remainder:]
    elif distribution == 'weighted':
        weights = distribution_args.get('weights')
        payload_chunks = weighted_chunks(payload, num_instances, weights)
    elif distribution == 'roundrobin':
        payload_chunks = roundrobin_chunks(payload, num_instances)
    else:
        raise NotImplementedError(f"Distribution {distribution} unknown")

    for instance, chunk in zip(instances, payload_chunks):
        output.append(instance(chunk))
    return output

    
def split_decode(instances, seed=None, distribution=None, distribution_args=None):
    """Decodes a payload across several instances.

       :param instances: The iterable of instances.
       :type instances: list
       :param seed: Seed to pseudo-randomly distribute the payload over the different instances.
       :type seed: int, optional
       :param distribution: How to distribute the payload across the instances.
       :type distribution: str, optional
       :param distribution_args: Additional distribution args
       :type distribution_args: dict, optional

       :return: The decoded payload
       :rtype: str
    """
    output = ''
    for instance in instances:
        output += instance()
    if distribution == 'roundrobin':
        output = reverse_roundrobin(output, len(instances))
        
    if seed is None:
        return output
    else:
        indices = utils.prng_indices(len(output),seed)
        zeros = np.zeros(len(output),dtype=int)
        for bit, index in zip(output, indices):
            zeros[index] = bit
        return ''.join(map(str, zeros))
