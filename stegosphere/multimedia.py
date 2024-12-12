from numpy import random
import numpy as np

from .utils import file_to_binary, data_to_binary
from . import utils

__all__ = ['split_encode','split_decode']

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
    if seed:
        indices = utils.prng_indices(len(message), seed)
        message = ''.join(message[i] for i in indices)
    else:
        indices = np.arange(len(message))

    message_chunks = []
    if distribution == 'even':
        chunk_length = len(message)//len(instances)
        for i in range(len(instances)):
            message_chunks.append(message[i*chunk_length:(i+1)*chunk_length])
        if remainder := len(message) % len(instances):
            message_chunks[-1] += message[-remainder:]
    else:
        raise NotImplementedError()

    for instance, chunk in zip(instances, message_chunks):
        instance(chunk)

    
def split_decode(instances, seed=None, distribution='even', distribution_args=None):
    """Decodes a message across several instances.

       :param instances: The iterable of instances.
       :type instances: list
       :param seed: Seed to pseudo-randomly distribute the message over the different instances.
       :type seed: int, optional
       :param distribution: How to distribute the message across the instances. Defaults to even distribution.
       :type distribution: str, optional
       :param distribution_args: Additional distribution args
       :type distribution_args: dict, optional

       :return: The decoded message
       :rtype: str
    """
    assert distribution=='even'
    output = ''
    for instance in instances:
        output += instance()
    if seed is None:
        return output
    indices = utils.prng_indices(len(output),seed)
    zeros = np.zeros(len(output),dtype=int)
    for bit, index in zip(output, indices):
        zeros[index] = bit
    return ''.join(map(str, zeros))
