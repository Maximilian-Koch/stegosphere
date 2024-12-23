import numpy as np
import warnings

from analysis import Analysis
import utils
import core

#include option for looping on RGB(A) images!
class BaseIWT(core.TransformMethod):
    def __init__(self, data, analysis=utils.ANALYSIS):
        assert type(data) == np.ndarray
        self._analysis = analysis
        if self._analysis:
            self.analysis = Analysis(data.copy())
        
        self.data = data
        self.dim = np.ndim(self.data)
    def _adjust_for_uneven_lengths(self, array):
        """
        Removes last element in uneven dimensions.

        Return array with even lengths and dictionary with removed element for each dimension.
        """
        adjusted_array = array.copy()
        removed_elements = {}

        for axis in reversed(range(array.ndim)):
            dim_size = adjusted_array.shape[axis]
            if dim_size % 2 == 0:
                continue
            else:
                slices = [slice(None)] * adjusted_array.ndim
                slices[axis] = slice(0, -1)
                
                last_element_slices = [slice(None)] * adjusted_array.ndim
                last_element_slices[axis] = slice(-1, None)
                removed_elements[axis] = adjusted_array[tuple(last_element_slices)]
                
                adjusted_array = adjusted_array[tuple(slices)]
        
        return adjusted_array, removed_elements
    
    def transform(self, loop_3d=False):
        """
        Transform data into wavelet domain.
        """
        self.data, self._adjust = self._adjust_for_uneven_lengths(self.data)
        self.data = self._iwt_nd(self.data)
        self.bands = list(self.data.keys())
        self.shapes = dict()
        for key in self.bands:
            self.shapes.update({key : self.data[key].shape})

    def inverse(self):
        """
        Inverse transform data into spatial domain.
        """
        self.data = self._iiwt_nd(self.data)
        self.data = self._restore_uneven_lengths(self.data, self._adjust)

        if self._analysis:
            self.analysis.after = self.data.reshape(self.analysis.before.shape)
            
    def __setitem__(self, key, value):
        if value.shape != self.shapes[key]:
            value = value.reshape(self.shapes[key])
        self.data[key] = value
        
    def __getitem__(self, item):
        try:
            return self.data[item]
        except (KeyError, IndexError):
            raise Exception('item not found. Make sure to run .transform() first.')

    def _restore_uneven_lengths(self, array, removed_elements):
        """
        Restore adjusted array.
        """
        restored_array = array.copy()
        for axis in sorted(removed_elements.keys()):
            elements = removed_elements[axis]
            restored_array = np.concatenate([restored_array, elements], axis=axis)
        return restored_array


    def _iwt_nd(self, array):
        """
        Perform an n-dimensional integer Haar wavelet transform.

        :param array: n-dimensional integer array
        :type array: np.ndarray

        :return: Dictionary with coefficients. Keys are tuples, with 0 for approximation and 1 for detail.
        :rtype: dict
        """
        array = array.astype(np.int64)

        if any(dim % 2 != 0 for dim in array.shape):
            raise ValueError("All dimensions of the input array must be even.")

        coeffs = {(): array}

        for axis in range(array.ndim):
            new_coeffs = {}
            for key, arr in coeffs.items():
                s = [slice(None)] * arr.ndim

                s_even = s.copy()
                s_even[axis] = slice(None, None, 2)
                arr_even = arr[tuple(s_even)]

                s_odd = s.copy()
                s_odd[axis] = slice(1, None, 2)
                arr_odd = arr[tuple(s_odd)]

                approx = (arr_even + arr_odd) // 2
                detail = arr_even - arr_odd

                key_approx = key + ('0',)
                key_detail = key + ('1',)

                new_coeffs[key_approx] = approx
                new_coeffs[key_detail] = detail

            coeffs = new_coeffs

        return coeffs

    def _iiwt_nd(self, coeffs):
        """
        Perform the inverse n-dimensional integer Haar wavelet transform.

        :param coeffs: Dictionary with coefficients.
        :param type: dict

        :return: Reconstructed array.
        :rtype: np.ndarray
        """
        ndim = len(next(iter(coeffs.keys())))  # Number of dimensions
        coeffs_inv = coeffs.copy()

        for axis in reversed(range(ndim)):
            new_coeffs = {}
            keys = set(k[:-1] for k in coeffs_inv.keys())

            for key in keys:
                key_approx = key + ('0',)
                key_detail = key + ('1',)

                approx = coeffs_inv[key_approx]
                detail = coeffs_inv[key_detail]

                s = [slice(None)] * approx.ndim
                shape = list(approx.shape)
                shape[axis] *= 2
                arr = np.zeros(shape, dtype=np.int64)

                s_even = s.copy()
                s_even[axis] = slice(None, None, 2)
                s_odd = s.copy()
                s_odd[axis] = slice(1, None, 2)

                arr_even = approx + ((detail + 1) // 2)
                arr_odd = arr_even - detail

                arr[tuple(s_even)] = arr_even
                arr[tuple(s_odd)] = arr_odd

                new_coeffs[key] = arr

            coeffs_inv = new_coeffs
        return coeffs_inv[()]
