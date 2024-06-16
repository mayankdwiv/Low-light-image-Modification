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
    

    bool_value = file_name[-3:] in ['bmp', 'jpg', 'png', 'tif']

    return bool_value
