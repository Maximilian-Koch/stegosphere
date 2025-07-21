"""
Implements HILL cost, as presented in

B. Li, M. Wang, J. Huang and X. Li,
"A new cost function for spatial image steganography," 
2014 IEEE International Conference on Image Processing (ICIP), Paris, France, 2014, pp. 4206-4210, doi: 10.1109/ICIP.2014.7025854.
"""

import numpy as np
from scipy.signal import convolve2d

def hill_cost(cover, alpha=1, size=3):
   """
   Calculates the HILL cost of a given cover.

   Args:
      cover (numpy.ndarray): Cover to be evaluated
      alpha (float, optional): alpha to be used
      size (int, optional): kernel size
   Returns:
      numpy.ndarray: Cost map
   """
    #high-pass kernels
   k1 = np.array([[-1, 2, -1]])
   k2 = k1.T
   #combined high-pass response
   hp = np.abs(convolve2d(cover, k1, mode='same')) \
      + np.abs(convolve2d(cover, k2, mode='same'))
   #low-pass smoothing
   lp = convolve2d(hp, np.ones((size,size))/pow(size,2), mode='same')
   #cost map (avoid divide-by-zero)
   return 1.0 / (lp + alpha)