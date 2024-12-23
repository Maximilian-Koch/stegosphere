from numpy import random
import numpy as np

import audio, image, ttf
from utils import file_to_binary, data_to_binary
import utils

__all__ = ['roundrobin_chunks', 'split_encode','split_decode', 'weighted_chunks']

def weighted_chunks(message, num_instances, weights):
    """
    Splits message into num_instances chunks according to weights.
    """
    assert sum(weights)==1
    assert len(weights)==num_instances

    length = len(message)
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
        chunks.append(message[start:end])
        start = end
        
    return chunks


def roundrobin_chunks(message, num_instances):
    """
    Distribute message in a cycle / round-robin.
    """
    rosters = [[] for _ in range(num_instances)]
    for i, bit in enumerate(message):
        rosters[i % num_instances].append(bit)

    return [''.join(chunk) for chunk in rosters]

def reverse_roundrobin(message, num_instances):
    """
    Reconstruct message from round-robin message.
    
    :param message: The entire string distributed using round-robin.
    :type message: str
    :param num_instances: The number of instances used
    :type num_instances: int
    
    :return: Reconstructed message.
    :rtype: str
    """
    chunk_lengths = [len(message) // num_instances] * num_instances
    remainder = len(message) % num_instances
    
    for i in range(remainder):
        chunk_lengths[i] += 1
    
    chunks = []
    start = 0
    for length in chunk_lengths:
        chunks.append(message[start:start + length])
        start += length
    

    output = [''] * len(message)
    positions = [0] * num_instances
    for i in range(len(message)):
        chunk_index = i % num_instances
        output[i] = chunks[chunk_index][positions[chunk_index]]
        positions[chunk_index] += 1
    
    return ''.join(output)




def split_encode(message, instances, seed=None, distribution='even', distribution_args=None):
    """Encodes a message across several instances.

       :param message: The message to be encoded.
       :type message: str
       :param instances: The iterable of instances.
       :type instances: list
       :param seed: Seed to pseudo-randomly distribute the message over the different instances.
       :type seed: int, optional
       :param distribution: How to distribute the message across the instances. Defaults to even distribution.
       :type distribution: str, optional
       :param distribution_args: Additional distribution args
       :type distribution_args: dict, optional
    """
    if distribution_args is None:
        distribution_args = {}
    
    if seed:
        indices = utils.prng_indices(len(message), seed)
        message = ''.join(message[i] for i in indices)
    else:
        indices = np.arange(len(message))

    num_instances = len(instances)
    message_chunks = []
    
    if distribution == 'even':
        chunk_length = len(message)//num_instances
        for i in range(len(instances)):
            message_chunks.append(message[i*chunk_length:(i+1)*chunk_length])
        if remainder := len(message) % len(instances):
            message_chunks[-1] += message[-remainder:]
    elif distribution == 'weighted':
        weights = distribution_args.get('weights')
        message_chunks = weighted_chunks(message, num_instances, weights)
    elif distribution == 'roundrobin':
        message_chunks = roundrobin_chunks(message, num_instances)
    else:
        raise NotImplementedError(f"Distribution {distribution} unknown")

    for instance, chunk in zip(instances, message_chunks):
        instance(chunk)

    
def split_decode(instances, seed=None, distribution=None, distribution_args=None):
    """Decodes a message across several instances.

       :param instances: The iterable of instances.
       :type instances: list
       :param seed: Seed to pseudo-randomly distribute the message over the different instances.
       :type seed: int, optional
       :param distribution: How to distribute the message across the instances.
       :type distribution: str, optional
       :param distribution_args: Additional distribution args
       :type distribution_args: dict, optional

       :return: The decoded message
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
