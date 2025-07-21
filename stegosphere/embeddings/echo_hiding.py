"""
Echo hiding steganography
"""

import numpy as np
from scipy.signal import fftconvolve

def echo_kernel(delay, alpha, length):
    """Generate an echo impulse-response kernel.

    Args:
        delay (int): Number of samples to delay the echo.
        alpha (float): Attenuation factor for the echo amplitude.
        length (int): Total length of the kernel in samples.

    Returns:
        np.ndarray: 1D kernel of shape `(length,)` with a direct-path impulse
            at index 0 and an echo impulse at index `delay`.
    """
    h = np.zeros(length)
    h[0] = 1.0
    h[delay] = alpha
    return h

def embed(cover,bits,frame_size= 1024,delay0= 100,delay1= 200,alpha = 0.3):
    """Embed a bitstring into audio using echo hiding.

    Splits the input signal into non-overlapping frames and applies either a
    short (`delay0`) or long (`delay1`) echo to encode '0' or '1' respectively.

    Args:
        cover (np.ndarray): Input audio signal. May be 1D (mono) or 2D
            with shape `(n_samples, n_channels)`.
        bits (str): Sequence of '0'/'1' characters to embed.
        frame_size (int, optional): Number of samples per bit frame.
            Defaults to 1024.
        delay0 (int, optional): Echo delay in samples to encode a '0' bit.
            Defaults to 100.
        delay1 (int, optional): Echo delay in samples to encode a '1' bit.
            Defaults to 200.
        alpha (float, optional): Echo amplitude relative to the direct signal.
            Defaults to 0.3.

    Returns:
        np.ndarray: Stego signal of same shape as `cover`, with embedded echoes.
    """
    mono = cover.mean(axis=1) if cover.ndim == 2 else cover
    n_frames = len(bits)
    out = mono.copy().astype(np.float32)
    for i, b in enumerate(bits):
        start = i * frame_size
        end = start + frame_size
        if end > len(mono):
            break
        frame = mono[start:end]
        d = delay0 if b == '0' else delay1
        h = echo_kernel(d, alpha, len(frame))
        echoed = fftconvolve(frame, h, mode='full')[:len(frame)]
        out[start:end] = echoed
    return out

def detect_delay(frame,search_delays, alpha):
    """Detect which echo delay is present in an audio frame.

    Performs autocorrelation of the frame and compares the peaks at specified
    delays to decide which echo was embedded.

    Args:
        frame (np.ndarray): 1D audio frame.
        search_delays (tuple of int): Candidate delays (e.g., `(delay0, delay1)`).
        alpha (float): Expected echo attenuation factor (unused in detection
            but included for API consistency).

    Returns:
        int: The delay value from `search_delays` that yields the highest
            autocorrelation peak, indicating the embedded bit.
    """
    # autocorrelation
    ac = np.correlate(frame, frame, mode='full')
    mid = len(ac) // 2
    # find which delay gives higher correlation peak
    scores = []
    for d in search_delays:
        idx = mid + d
        scores.append(ac[idx])
    return search_delays[int(np.argmax(scores))]

def extract(stego, n_bits, frame_size = 1024,delay0 = 100,delay1= 200,alpha = 0.3):
    """Extract a bitstring from stego audio using echo hiding.

    Splits the stego signal into frames and detects which delay was used to
    encode each bit.

    Args:
        stego (np.ndarray): Stego audio signal, mono or `(n_samples, n_channels)`.
        n_bits (int): Number of bits to extract.
        frame_size (int, optional): Number of samples per bit frame.
            Defaults to 1024.
        delay0 (int, optional): Echo delay used to encode '0'.
            Defaults to 100.
        delay1 (int, optional): Echo delay used to encode '1'.
            Defaults to 200.
        alpha (float, optional): Echo attenuation factor used in embedding.
            Defaults to 0.3.

    Returns:
        str: Extracted bitstring of length up to `n_bits`.
    """
    stego = stego.mean(axis=1) if stego.ndim == 2 else stego
    bits = []
    for i in range(n_bits):
        start = i * frame_size
        end = start + frame_size
        if end > len(stego):
            break
        frame = stego[start:end]
        d = detect_delay(frame, (delay0, delay1), alpha)
        bits.append('0' if d == delay0 else '1')
    return ''.join(bits)

