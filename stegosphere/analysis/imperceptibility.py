"""
Measuring imperceptibility
"""

import numpy as np

from stegosphere import utils

__all__ = ['mse','psnr','ssim','manhattan_distance','euclidean_distance',
        'chi_distance','kl_divergence','js_divergence',
        'bhattacharyya_divergence','results']

def mse(array_1, array_2):
    """
    Compute Mean Squared Error (MSE) between two arrays.

    Args:
        array_1, array_2 (array-like): Input arrays of equal shape.

    Returns:
        float: MSE value.

    Raises:
        ValueError: If shapes differ.
    """
    a1 = np.asarray(array_1, dtype=np.float64)
    a2 = np.asarray(array_2, dtype=np.float64)
    if a1.shape != a2.shape:
        raise ValueError("Input arrays must have the same shape")
    return np.mean((a1 - a2) ** 2)

def psnr(array_1, array_2, max_i=None):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels.

    Args:
        array_1, array_2 (array-like): Input arrays of equal shape.
        max_i (number, optional): Maximum possible pixel value.
            Defaults to dtype max of array_1.

    Returns:
        float: PSNR value (inf if zero error).

    Raises:
        ValueError: If shapes differ.
    """
    mse_val = mse(array_1, array_2)
    if mse_val == 0:
        return float('inf')
    if max_i is None:
        max_i = utils.dtype_range(np.asarray(array_1).dtype)[1]
    return 10 * np.log10((max_i ** 2) / mse_val)
    

def ssim(array_1, array_2, **kwargs):
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        array_1, array_2 (ndarray): Grayscale or multi-channel images.
        **kwargs: Passed to skimage.metrics.structural_similarity.

    Returns:
        float or ndarray: SSIM score (and image if full=True).

    Raises:
        ValueError: If shapes differ.
    """
    from skimage.metrics import structural_similarity as _ssim
    a1 = np.asarray(array_1)
    a2 = np.asarray(array_2)
    if a1.shape != a2.shape:
        raise ValueError("Input images must have the same shape")
    return _ssim(a1, a2, **kwargs)


def manhattan_distance(array_1, array_2):
    """
    Compute Manhattan (L1) distance between two arrays.

    Args:
        array_1, array_2 (array-like): Input arrays of equal shape.

    Returns:
        float: Sum of absolute differences.

    Raises:
        ValueError: If shapes differ.
    """
    a1 = np.asarray(array_1)
    a2 = np.asarray(array_2)
    if a1.shape != a2.shape:
        raise ValueError("Input arrays must have the same shape")
    return np.sum(np.abs(a1 - a2))

def euclidean_distance(array_1, array_2):
    """
    Compute Euclidean (L2) distance between two arrays.

    Args:
        array_1, array_2 (array-like): Input arrays of equal shape.

    Returns:
        float: Euclidean norm of differences.

    Raises:
        ValueError: If shapes differ.
    """
    a1 = np.asarray(array_1, dtype=np.float64)
    a2 = np.asarray(array_2, dtype=np.float64)
    if a1.shape != a2.shape:
        raise ValueError("Input arrays must have the same shape")
    return np.linalg.norm(a1 - a2)

def chi_distance(array_1, array_2, epsilon=1e-10):
    """
    Compute chi-squared distance between two nonnegative arrays.

    Args:
        array_1, array_2 (array-like): Input histograms or nonnegative arrays.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: Chi-squared distance.

    Raises:
        ValueError: If shapes differ.
    """
    a1 = np.asarray(array_1, dtype=np.float64)
    a2 = np.asarray(array_2, dtype=np.float64)
    if a1.shape != a2.shape:
        raise ValueError("Input arrays must have the same shape")
    num = (a1 - a2) ** 2
    den = a1 + a2 + epsilon
    return np.sum(num / den)


def kl_divergence(array_1, array_2, epsilon=1e-10):
    """
    Compute Kullback-Leibler divergence Dₖₗ(P‖Q).

    Args:
        array_1, array_2 (array-like): Probability distributions or counts.
        epsilon (float): Small constant to avoid log(0).

    Returns:
        float: KL divergence.

    Raises:
        ValueError: If shapes differ.
    """
    p = np.asarray(array_1, dtype=np.float64)
    q = np.asarray(array_2, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape")
    p /= (p.sum() + epsilon)
    q /= (q.sum() + epsilon)
    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))


def js_divergence(array_1, array_2, epsilon=1e-10):
    """
    Compute Jensen-Shannon divergence between two distributions.

    Args:
        array_1, array_2 (array-like): Probability distributions or counts.
        epsilon (float): Small constant to avoid log(0).

    Returns:
        float: JS divergence.

    Raises:
        ValueError: If shapes differ.
    """
    p = np.asarray(array_1, dtype=np.float64)
    q = np.asarray(array_2, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape")
    p /= (p.sum() + epsilon)
    q /= (q.sum() + epsilon)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log((p + epsilon) / (m + epsilon))) +
                  np.sum(q * np.log((q + epsilon) / (m + epsilon))))

def bhattacharyya_divergence(array_1, array_2, epsilon=1e-10):
    """
    Compute Bhattacharyya distance between two distributions.

    Args:
        array_1, array_2 (array-like): Probability distributions or counts.
        epsilon (float): Small constant to avoid log(0).

    Returns:
        float: Bhattacharyya distance.

    Raises:
        ValueError: If shapes differ.
    """
    p = np.asarray(array_1, dtype=np.float64)
    q = np.asarray(array_2, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape")
    p /= (p.sum() + epsilon)
    q /= (q.sum() + epsilon)
    bc = np.sum(np.sqrt(p * q))
    return -np.log(bc + epsilon)

def results(array_1, array_2, measures='all', epsilon=1e-10, psnr_max_i=None):
    """
    Compute multiple similarity/divergence/distance measures.

    Args:
        array_1, array_2 (array-like): Input arrays of equal shape.
        measures (list or 'all'): Names of measures to compute.
        epsilon (float, optional): Small constant for stability.
        psnr_max_i (number, optional): Max value for PSNR.

    Returns:
        dict: {measure_name: value, ...}
    """
    if measures == 'all' :
        measures = ['mse','psnr','ssim','kl','js','bhattacharyya','manhattan','euclidean','chi']
    #mse, psnr, ssim, kl, js, bhattacharyya, manhattan, euclidean, chi

    output = {}
    if 'psnr' in measures or 'mse' in measures:
        mse = np.mean((self.before.astype(np.float64) - self.after.astype(np.float64)) ** 2)
        if 'mse' in measures:
            output.update({'mse' : mse})
    if 'psnr' in measures:
        if mse == 0:
            psnr = float('inf')
        else:
            if psnr_max_i is None:
                psnr_max_i = utils.dtype_range(self.before.dtype)[1]
            psnr = 10 * np.log10((psnr_max_i ** 2) / mse)
        output.update({'psnr' : psnr})
        
    if 'ssim' in measures:
        pass
    if 'kl' in measures or 'js' in measures or 'bhattacharyya' in measures:
        p,q = array_1, array_2

        #normalize
        p = p.astype(np.float64)
        q = q.astype(np.float64)
        p /= (p.sum() + epsilon)
        q /= (q.sum() + epsilon)
        if 'kl' in measures:
        #Kullback-Leibler divergence
            kl = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
            output.update({'kl':kl})
        if 'js' in measures:
            #Jensen-Shannon divergence
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
            kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
            js = 0.5 * (kl_pm + kl_qm)
            output.update({'js':js})
        if 'bhattacharyya' in measures:
            bc = np.sum(np.sqrt(p*q))
            ba = -np.log(bc+epsilon)
            output.update({'bhattacharyya' : ba})
    if 'manhattan' in measures or 'euclidean' in measures or 'chi' in measures:
        diff = array_1 - array_2
        if 'euclidean' or 'chi' in measures:
            diffsquare = diff**2
            if 'euclidean' in measures:
                euclidean = np.sqrt(np.sum(diffsquare))
                output.update({'euclidean' : euclidean})
            if 'chi' in measures:
                chi = np.sum(diffsquare / (array_1+array_2+epsilon))
                output.update({'chi' : chi})
        if 'manhattan' in measures:
            manhattan = np.sum(np.abs(diff))
            output.update({'manhattan' : manhattan})
            
    return output
