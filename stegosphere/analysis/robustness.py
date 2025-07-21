"""
A module for simulating attacks onto arrays.
"""

import numpy as np
import cv2

from stegosphere.utils import dtype_range

__all__ = ['random_bit_attack', 'gaussian_noise_attack', 'jpeg_compression_attack', 'cropping_attack']

def random_bit_attack(array, ratio=0.01):
    """
    Randomly flip a given fraction of elements in the array by +1 or -1.

    Args:
        array (array-like): Input numeric array.
        ratio (float): Fraction of elements to perturb ∈ [0, 1].

    Returns:
        np.ndarray: Copy of the input with random ±1 perturbations applied.

    Raises:
        ValueError: If ratio is not between 0 and 1.
    """
    if not (0 <= ratio <= 1):
        raise ValueError(f"ratio must be between 0 and 1; got {ratio}")

    arr = np.asarray(array)
    n = arr.size
    k = int(n * ratio)
    if k == 0:
        return arr.copy()

    out = arr.copy()
    indices = np.random.choice(n, size=k, replace=False)
    perturb = np.random.choice([-1, 1], size=k)

    out_flat = out.flat
    for idx, delta in zip(indices, perturb):
        out_flat[idx] = out_flat[idx] + delta

    if np.issubdtype(out.dtype, np.integer):
        min_val, max_val = dtype_range(out.dtype)
        out = np.clip(out, min_val, max_val)

    return out

def gaussian_noise_attack(array, sigma=1.0):
    """
    Apply Gaussian smoothing as a low-pass filter.

    Args:
        array (ndarray): Input image or signal.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        ndarray: Smoothed result.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(array, sigma=sigma)


def salt_and_pepper_attack(image, amount=0.005, salt_vs_pepper=0.5):
    """
    Add salt-and-pepper noise to an image.

    Args:
        image (ndarray): Input image.
        amount (float): Proportion of pixels to alter.
        salt_vs_pepper (float): Ratio of salt vs. pepper noise.

    Returns:
        ndarray: Noisy image.
    """
    out = np.copy(image)
    total = image.size
    num_salt = int(total * amount * salt_vs_pepper)
    num_pepper = int(total * amount * (1 - salt_vs_pepper))

    coords = [np.random.randint(0, d, num_salt) for d in image.shape]
    out[tuple(coords)] = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1
    coords = [np.random.randint(0, d, num_pepper) for d in image.shape]
    out[tuple(coords)] = np.iinfo(image.dtype).min if np.issubdtype(image.dtype, np.integer) else 0
    return out


def jpeg_compression_attack(image, quality=50):
    """
    Simulate JPEG compression artifacts.

    Args:
        image (ndarray): Input image in BGR or grayscale.
        quality (int): JPEG quality factor (0-100).

    Returns:
        ndarray: Compressed-then-decoded image.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encimg = cv2.imencode('.jpg', image, encode_param)
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decimg = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    return decimg


def cropping_attack(image, crop_fraction=0.1):
    """
    Randomly crop and pad back to original size.

    Args:
        image (ndarray): Input image.
        crop_fraction (float): Fractional size to crop on each side.

    Returns:
        ndarray: Cropped and resized image.
    """
    h, w = image.shape[:2]
    dh, dw = int(h * crop_fraction), int(w * crop_fraction)
    top = np.random.randint(0, dh)
    left = np.random.randint(0, dw)
    cropped = image[top:h-dh, left:w-dw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
