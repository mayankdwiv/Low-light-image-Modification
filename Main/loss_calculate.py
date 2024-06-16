import numpy as np
def loss_calculate(
        reference_image: np.ndarray,
        refined_image: np.ndarray
        ) -> float:
    """Calculates the lightness order error (LOE) metric comparing pixel 
    intensities of a refined image with their reference counterparts.

    Returns a calculated value of the LOE metric.

    ## Args:
        reference_image (numpy.ndarray) : A shape-(3, M, N) reference image 
        which is considered as ground truth.

        refined_image (numpy.ndarray) : A shape-(3, M, N) refined image.

    ## Returns:
        (float) : A calculated value of the LOE metric.
    """

    v_shape, h_shape = reference_image.shape
    n_pixels = reference_image.size
    loss = 0

    for v_pixel in range(v_shape-1):
        for h_pixel in range(h_shape-1):
            bool_term_ini = reference_image <= \
                  reference_image[v_pixel, h_pixel]
            bool_term_ref = refined_image <= refined_image[v_pixel, h_pixel]
            xor_term = np.logical_xor(bool_term_ini, bool_term_ref)
            loss += np.sum(xor_term)

    return loss / (n_pixels * 1000)