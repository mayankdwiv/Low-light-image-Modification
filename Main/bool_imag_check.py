import cv2
import numpy as np
import os as os
from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d
from typing import Union
def bool_image(file_name: str) -> bool:
    """Checks if a file is of 'bmp', 'jpg', 'png' or 'tif' format.

    Returns True if a file name ends with any of these formats, and False is 
    returned otherwise.

    ## Args:
        file_name (str) : A string representing a file name.

    ## Returns:
        bool_value (bool) : A boolean value answering if the provided file 
        name is of the given four formats.   
    """

    bool_value = file_name[-3:] in ['bmp', 'jpg', 'png', 'tif']

    return bool_value