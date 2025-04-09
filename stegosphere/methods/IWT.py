import numpy as np
import warnings

MODE_DISCARD = 'discard'
MODE_SYMMETRIC = 'symmetric'
MODE_REFLECT = 'reflect'
MODE_EDGE = 'edge'
MODE_WRAP = 'wrap'

SUPPORTED_PAD_MODES = [MODE_DISCARD,MODE_SYMMETRIC,MODE_REFLECT,MODE_EDGE,MODE_WRAP]

def transform(array, skip_last_axis=False, boundary_mode=MODE_DISCARD):
    """
    Transform data into wavelet domain using the Integer Haar Wavelet Transform.
    Uneven dimensions are handled using skip_last_axis and boundary mode.

    :param array: input numpy array
    :param skip_last_axis: If True, the transformation is not applied along the last axis. Recommended for colour images.
    :param boundary_mode: Method to handle uneven dimensions along axes being transformed.
                        Options:
                        - 'discard' : remove the last slice and restore later
                        - 'symmetric'/'reflect'/'edge'/'wrap': use np.pad to make dimensions even,
                        and remove padding after inverse transform
    :return: A tupling containing:
                        - dictionary with wavelet coefficients
                        - information needed for the inverse transform
    """
    
    if not isinstance(array, np.ndarray):
        raise TypeError('array must be a numpy ndarray')

    boundary_info = {'mode': boundary_mode}

    if boundary_mode == MODE_DISCARD:
        processed_array, removed_elements = _adjust_for_uneven_lengths(array, skip_last_axis)
        boundary_info['details'] = removed_elements # Store removed slices
    elif boundary_mode in SUPPORTED_PAD_MODES:
        warnings.warn('boundary modes other than discard are not stable yet.')
        processed_array, padded_axes = _pad_for_even_lengths(array, skip_last_axis, boundary_mode)
        boundary_info['details'] = padded_axes # Store which axes were padded
    else:
        raise ValueError('unsupported boundary mode')

    coeffs = _iwt_nd(processed_array, skip_last_axis)
    return coeffs, boundary_info

def inverse(coeffs, boundary_info):
    """
    Inverse transform data from wavelet domain into spatial domain using teh Integer Haar Wavelet Transform.

    :param coeffs: Dictionary with wavelet coefficients from the transform function
    :param boundary_info: Dictionary containing the boundary handling information as returned by the transform function

    :return: Reconstructed numpy array
    """
    if not isinstance(coeffs, dict):
        raise TypeError('coeffs must be a dictionary.')
    if not isinstance(boundary_info, dict):
        raise TypeError('boundary_info must be a dictionary.')
    if 'mode' not in boundary_info or 'details' not in boundary_info:
        raise KeyError('boundary_info dictionary must contain "mode" and "details" keys.')

    mode = boundary_info['mode']
    details = boundary_info['details']

    reconstructed_processed_array = _iiwt_nd(coeffs)
    if mode == MODE_DISCARD:
        #details contain the removed slices
        final_array = _restore_uneven_lengths(reconstructed_processed_array, details)
    elif mode in SUPPORTED_PAD_MODES:
        #details contain the axes that were padded
        final_array = _remove_padding(reconstructed_processed_array, details)
    return final_array


def _adjust_for_uneven_lengths(array, skip_last_axis=False):
    """adjusting for discard mode"""
    adjusted_array = array.copy()
    removed_elements = {} # Store {axis: removed_slice_array}

    num_dims_to_process = array.ndim
    if skip_last_axis and array.ndim > 0:
        num_dims_to_process -= 1

    for axis in range(num_dims_to_process):
        dim_size = adjusted_array.shape[axis]
        if dim_size % 2 == 0 or dim_size == 0:
            continue
        else: # dim_size is odd and > 0
            slices_keep = [slice(None)] * adjusted_array.ndim
            slices_keep[axis] = slice(0, -1)
            slices_remove = [slice(None)] * adjusted_array.ndim
            slices_remove[axis] = slice(-1, None)

            removed_elements[axis] = adjusted_array[tuple(slices_remove)]
            adjusted_array = adjusted_array[tuple(slices_keep)]

    return adjusted_array, removed_elements

def _restore_uneven_lengths(array, removed_elements):
    restored_array = array.copy()
    if not removed_elements:
        return restored_array
    for axis in sorted(removed_elements.keys(), reverse=True):
        elements_to_add = removed_elements[axis]
        current_shape = list(restored_array.shape)
        expected_element_shape = current_shape[:]
        expected_element_shape[axis] = 1
        element_shape = list(elements_to_add.shape)
        if tuple(element_shape) != tuple(expected_element_shape):
            raise ValueError(f'Shape mismatch for axis {axis}. Expected {expected_element_shape}, got {element_shape}')
        restored_array = np.concatenate([restored_array, elements_to_add], axis=axis)
    return restored_array

def _pad_for_even_lengths(array, skip_last_axis=False, pad_mode=MODE_SYMMETRIC):
    """padding uneven dimensions"""
    padded_array = array.copy()
    padded_axes = {} # Store {axis: pad_amount (always 1 here)}

    num_dims_to_process = array.ndim
    if skip_last_axis and array.ndim > 0:
        num_dims_to_process -= 1

    axes_needing_padding = []
    for axis in range(num_dims_to_process):
        if padded_array.shape[axis] % 2 != 0 and padded_array.shape[axis] > 0:
             axes_needing_padding.append(axis)
             padded_axes[axis] = 1 # Record axis and pad amount (1)

    if not axes_needing_padding:
        return padded_array, padded_axes

    pad_width = [(0, 0)] * padded_array.ndim
    for axis in axes_needing_padding:
         pad_width[axis] = (0, 1) 

    padded_array = np.pad(padded_array, pad_width, mode=pad_mode)

    return padded_array, padded_axes


def _remove_padding(array, padded_axes):
    """remove padding"""
    if not padded_axes:
        return array.copy()

    slices = [slice(None)] * array.ndim
    for axis, pad_amount in padded_axes.items():
        if pad_amount != 1:
             raise ValueError(f'Padding removal logic assumes pad_amount=1, but got {pad_amount} for axis {axis}')
        try:
            slices[axis] = slice(0, -1) # Remove last element added by padding
        except IndexError:
             raise IndexError(f'Axis {axis} from "padded_axes" is out of bounds '
                              f'for array shape {array.shape} during padding removal.')

    return array[tuple(slices)].copy()


def _iwt_nd(array, skip_last_axis=False):
    """
    Integer Haar Wavelet Transform. Assumes even dimensions
    """
    current_array = array.astype(np.int64)
    coeffs = {(): current_array}

    num_dims_to_process = array.ndim
    axes_to_process = list(range(array.ndim))
    if skip_last_axis and array.ndim > 0:
        num_dims_to_process -= 1
        axes_to_process = list(range(array.ndim - 1))

    for axis in axes_to_process:
         if array.shape[axis] % 2 != 0:
              raise ValueError('dimension has to be even')

    for axis in axes_to_process:
        new_coeffs = {}
        for key, arr_coeff in coeffs.items():
            s = [slice(None)] * arr_coeff.ndim
            s_even = s.copy(); s_even[axis] = slice(None, None, 2)
            s_odd = s.copy(); s_odd[axis] = slice(1, None, 2)
            arr_even = arr_coeff[tuple(s_even)]
            arr_odd = arr_coeff[tuple(s_odd)]

            approx = (arr_even + arr_odd) // 2
            detail = arr_even - arr_odd

            key_approx = key + ('0',)
            key_detail = key + ('1',)
            new_coeffs[key_approx] = approx
            new_coeffs[key_detail] = detail
        coeffs = new_coeffs

    return coeffs

def _iiwt_nd(coeffs):
    """
    Inverse Integer Haar Wavelet Transform. Assumes even dimensions
    """
    try:
        first_key = next(iter(coeffs.keys()))
        transformed_ndim = len(first_key)
    except StopIteration:
        if () in coeffs:
            return coeffs[()].astype(np.int64)
        else:
            raise ValueError('Coefficients dictionary is non-empty but contains no keys.')

    coeffs_inv = coeffs.copy()
    axes_to_invert = list(range(transformed_ndim))

    for axis in reversed(axes_to_invert):
        new_coeffs = {}
        processed_keys = set(k[:-1] for k in coeffs_inv.keys())

        for key_prefix in processed_keys:
            key_approx = key_prefix + ('0',)
            key_detail = key_prefix + ('1',)

            if key_approx not in coeffs_inv or key_detail not in coeffs_inv:
                raise ValueError(f'Missing pair for key prefix {key_prefix}.')

            approx = coeffs_inv[key_approx]
            detail = coeffs_inv[key_detail]
            current_ndim = approx.ndim

            if detail.ndim != current_ndim:
                 raise ValueError(f'Dimension mismatch between approx ({current_ndim}D) and detail ({detail.ndim}D) '
                                  f'for key prefix {key_prefix}')
            shape = list(approx.shape)
            if axis >= len(shape):
                 raise IndexError(f'Internal error: Axis index {axis} out of bounds for shape {approx.shape}.')

            shape[axis] *= 2
            arr = np.zeros(shape, dtype=np.int64)
            s = [slice(None)] * current_ndim
            s_even = s.copy(); s_even[axis] = slice(None, None, 2)
            s_odd = s.copy(); s_odd[axis] = slice(1, None, 2)

            #haar
            arr_even = approx + ((detail + 1) // 2)
            arr_odd = arr_even - detail

            if arr[tuple(s_even)].shape != arr_even.shape:
                 raise ValueError(f'Shape mismatch assigning even part for axis {axis}, key {key_prefix}. '
                                  f'Target: {arr[tuple(s_even)].shape}, Source: {arr_even.shape}')
            if arr[tuple(s_odd)].shape != arr_odd.shape:
                 raise ValueError(f'Shape mismatch assigning odd part for axis {axis}, key {key_prefix}. '
                                  f'Target: {arr[tuple(s_odd)].shape}, Source: {arr_odd.shape}')

            arr[tuple(s_even)] = arr_even
            arr[tuple(s_odd)] = arr_odd
            new_coeffs[key_prefix] = arr
        coeffs_inv = new_coeffs

    if () not in coeffs_inv:
         raise KeyError('Inverse transform failed: Final reconstructed array not found under key "()".')

    return coeffs_inv[()].astype(np.int64)
