#Detects the likelihood of steganography

import numpy as np
import random

def classifier_evaluation(detector, inputs, outputs):
    """
    Evaluates a steganalysis classifier.

    :param detector: detector function
    :type detector: callable
    :param inputs: inputs to classify
    :type inputs: iterable
    :param outputs: true outputs
    :type outputs: iterable
    """
    
    assert len(inputs)==len(outputs)
    t_p, t_n, f_p, f_n = 0,0,0,0
    for inp, out in zip(inputs, outputs):
        estimate = detector(inp)
        if estimate == 1 and out == 1: t_p += 1
        elif estimate == 1 and out == 0: f_p += 1
        elif estimate == 0 and out == 1: f_n += 1
        elif estimate == 0 and out == 0: t_n += 1

    return {'true_positive' : t_p, 'true_negative' : t_n,
            'false_positive' : f_p, 'false_negative' : f_n}

#For test purposes
def random_detector(input):
    return random.randint(0,1)


def uniformity_detector(array):
    """
    (for test purposes)
    Detects the likelihood of LSB steganography in an RGB image.
    """
    if len(array.shape) != 3 or array.shape[2] != 3:
        raise Exception("Input must be RGB image array.")
    
    #LSB of each channel
    lsb_red = array[:, :, 0] % 2
    lsb_green = array[:, :, 1] % 2
    lsb_blue = array[:, :, 2] % 2
    lsb_all = np.concatenate([lsb_red.flatten(), lsb_green.flatten(), lsb_blue.flatten()])

    count_zeros = np.sum(lsb_all == 0)
    count_ones = np.sum(lsb_all == 1)

    total = count_zeros + count_ones
    zero_ratio = count_zeros / total
    one_ratio = count_ones / total

    return 1 - abs(zero_ratio - one_ratio)

