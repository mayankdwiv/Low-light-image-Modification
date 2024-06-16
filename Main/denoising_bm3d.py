import cv2
import numpy as np

from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d
from typing import Union
def denoising_bm3d(
        image: np.ndarray,
        cor_ill_map: np.ndarray,
        std_dev: Union[int, float]=0.02
        ) -> np.ndarray:
    # """Performes denoising of an image Y color channel with B3MD algorithm and 
    # corrects its brigghtness with an updated illumination map.

    # Returns a shape-(M, N) denoised image with corrected brightness in which
    # pixel intensities exceeding 1 are clipped.

    # ## Args:
    #     image (numpy.ndarray) : A shape-(3, M, N) initial image.

    #     cor_ill_map (numpy.ndarray) : A shape-(M, N) array of 
    #     corrected intensity values.

    #     std_dev (int or float) : A value of standard deviation parameter for 
    #     the BM3D algorithm.

    # ## Returns:
    #     (numpy.ndarray) : A shape-(M, N) denoised image with a corrected 
    #     illumination map.
    # """

    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    y_channel = image_yuv[:, :, 0]
    denoised_y_ch = bm3d(y_channel, std_dev)
    image_yuv[:, :, 0] = denoised_y_ch
    denoised_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    recombined_image = image * cor_ill_map + denoised_rgb * (1 - cor_ill_map)

    return np.clip(recombined_image, 0, 1).astype("float32")
