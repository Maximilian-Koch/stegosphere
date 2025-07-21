"""
Module to visualize various measures.
Currently mostly for images.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = ['difference_heatmap','plot_lsb_frequency','channel_correlation_plot', 
          'histogram_comparison', 'show_side_by_side']


def difference_heatmap(orig, modified, title='Difference Heatmap', cmap='seismic', vmin=None, vmax=None):
    """
    Plot heatmap of differences between two arrays.

    Args:
        orig, modified (ndarray): Arrays of the same shape.
        title (str): Plot title.
        cmap (str): Colormap.
        vmin, vmax (float or None): Value range for color scaling.
    """
    orig = np.asarray(orig)
    modified = np.asarray(modified)
    if orig.shape != modified.shape:
        raise ValueError("Inputs must have the same shape")

    diff = modified.astype(np.float32) - orig.astype(np.float32)
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(diff, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    plt.show()


def plot_lsb_frequency(data):
    """
    Plot LSB frequency of an array (supports multichannel images, audio, etc.).

    Args:
        data (ndarray): Input array.
    """
    arr = np.asarray(data)
    if arr.ndim == 1:
        lsb = np.mod(arr, 2)
        counts = [np.mean(lsb)]
        labels = ['Signal']
    elif arr.ndim == 2:
        lsb = np.mod(arr, 2)
        counts = [np.mean(lsb)]
        labels = ['Matrix']
    elif arr.ndim == 3:
        lsb = np.mod(arr, 2)
        counts = [np.mean(lsb[..., i]) for i in range(arr.shape[2])]
        labels = [f'Channel {i}' for i in range(arr.shape[2])]
    else:
        raise ValueError("Unsupported array shape for LSB visualization")

    plt.figure()
    sns.barplot(x=labels, y=counts)
    plt.ylim(0, 1)
    plt.ylabel("Mean LSB Value")
    plt.title("LSB Frequency")
    plt.show()


def channel_correlation_plot(data):
    """
    Plot correlation between channels of a multichannel array.

    Args:
        data (ndarray): Multichannel array, last dim = channels.
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError("Expected 3D array with shape (H, W, C) or (T, N, C)")

    flat = arr.reshape(-1, arr.shape[-1])
    labels = [f'Channel {i}' for i in range(flat.shape[1])]
    df = {label: flat[:, i] for i, label in enumerate(labels)}
    sns.pairplot(sns.load_dataset("iris").assign(**df)[labels])
    plt.suptitle("Channel Correlation", y=1.02)
    plt.show()


def histogram_comparison(original, modified, bins=256):
    """
    Compare histograms of two arrays.

    Args:
        original, modified (ndarray): Input arrays.
        bins (int): Histogram bins.
    """
    a = np.asarray(original).ravel()
    b = np.asarray(modified).ravel()

    plt.figure(figsize=(10, 4))
    plt.hist(a, bins=bins, alpha=0.5, label='Original')
    plt.hist(b, bins=bins, alpha=0.5, label='Modified')
    plt.title('Histogram Comparison')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_side_by_side(a1, a2, titles=('Original', 'Modified')):
    """
    Show two arrays (e.g., images, spectrograms) side by side.

    Args:
        a1, a2 (ndarray): Arrays to show.
        titles (tuple): Subplot titles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, arr, title in zip(axes, [a1, a2], titles):
        ax.imshow(np.squeeze(arr), cmap='gray', aspect='auto')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
