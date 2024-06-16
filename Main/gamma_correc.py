import cv2
import numpy as np

from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d
from typing import Union
def gamma_correct(
        ill_map: np.ndarray,
        gamma: Union[int, float]
        ) -> np.ndarray:
    # """
    #     gamma (int or float) : A value of gamma correction coefficient.

    # ## Returns:
    #     (numpy.ndarray) : A shape-(M, N) array of corrected values of 
    #     the illumination map.
    # """

    return ill_map ** gamma
