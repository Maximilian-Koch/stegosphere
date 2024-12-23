import numpy as np


import utils
from utils import generate_binary_payload

__all__ = ['Analysis', 'generate_binary_payload', 'image_method_analysis']

class Analysis:
    def __init__(self, before, after=None):
        """
        :param before: The array before encoding a message.
        :type before: np.ndarray
        :param after: The array after encoding a message. Gets set automatically when using .encode()
        :type after: np.ndarray
        """
        self.before = before
        self.after = after
    def results(self):
        return {'mse' : self.mse(), 'psnr' : self.psnr()}
    def mse(self):
        """
        Calculates the Mean Squared Error (MSE) between the original and encoded array.

        :return: MSE value
        :rtype: float
        """
        assert self.after is not None
        return np.mean((self.before.astype(np.float64) - self.after.astype(np.float64)) ** 2)
    def psnr(self):
        """
        Calculates the Peak Signal-to-Noise Ratio (PSNR) between the original and encoded array.
        
        :return: PSNR value in dB.
        :rtype: float
        """
        assert self.before.shape == self.after.shape, "Arrays must have the same shape."
        mse_val = self.mse()
        if mse_val == 0:
            return float('inf')
        max_i = utils.dtype_range(self.before.dtype)[1]
        return 10 * np.log10((max_i ** 2) / mse_val)
    def ssim(self):
        """
        Calculates the Structural Index Similarity (SSIM) between the original and encoded array.
        """
        from skimage.metrics import structural_similarity as ssim
        assert self.after is not None
        return ssim(self.before, self.after)


def image_method_analysis(method, files, payload):
    """
    Analyses a method using steganalysis.

    :param method: A class containing an encoding method for arrays.
    :type method: class
    :param files: A list of files to be used for testing.
    :type files: list
    :param payload: A payload to be hidden in each of the files.
    :type payload: str

    :return: Data frame containing results.
    :rtype: pandas.DataFrame
    """
    import pandas as pd
    df = pd.DataFrame(columns=['mse','psnr'])
    for file in files:
        m = method(file)
        m.encode(payload)
        df.loc[file] = m.analysis.mse(), m.analysis.psnr()
    return df
