import cv2
import numpy as np

from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d
from typing import Union

def d_sparse_matrices(illumination_map: np.ndarray) -> csr_matrix:
    # """Generates Toeplitz matrices of the compatible shape with the given 
    # ''illumination_map''
    # for computation of a forward difference in both horizontal and vertical 
    # directions.

    # Returns the shape-(M*N, M*N) arrays of Toeplitz matrices in a compressed 
    # sparse row format.

    # ## Args:
    #     illumination_map (numpy.ndarray) : A shape-(M, N) array of maximum 
    #     intensity values.

   

    #     row2_sparsed (scipy.sparse.csr_matrix) : A shape-(M*N, M*N) compressed 
    #     sparse row matrix for calculation of a forward difference 
    #     in a vertical direction.
    # """

    image_x_shape = illumination_map.shape[-1]
    image_size = illumination_map.size
    dx_row, dx_col, dx_value = [], [], []
    dy_row, dy_col, dy_value = [], [], []
    # Produces lists of non-zero values and their row and column indeces
    for i in range(image_size - 1):
        if image_x_shape + i < image_size:
            dy_row += [i, i]
            dy_col += [i, image_x_shape + i]
            dy_value += [-1, 1]
        if (i+1) % image_x_shape != 0 or i == 0:
            dx_row += [i, i]
            dx_col += [i, i+1]
            dx_value += [-1, 1]
    # Creates compressed sparse row matrices of a required shape
    # based on provided values and their indeces
    row_sparsed = csr_matrix((dx_value, (dx_row, dx_col)), 
                            shape = (image_size, image_size))
    row2_sparsed = csr_matrix((dy_value, (dy_row, dy_col)), 
                            shape = (image_size, image_size))

    return row_sparsed, row2_sparsed


def partial_derivative_vectorized(
        input_matrix: np.ndarray,
        toeplitz_sparse_matrix: csr_matrix
        ) -> np.ndarray:
    # """Calculates a partial derivative of an ''input_matrix'' with a given 
    # ''toeplitz_sparse_matrix''.

    # Returns the shape-(M, N) array of derivative values.

    

    # ## Returns:
    #     p_derivative (numpy.ndarray) : A shape-(M, N) array of derivative 
    #     values.
    # """

    input_size = input_matrix.size
    output_shape = input_matrix.shape
    # Vectorizes the input matrix producing a shape-(M*N, 1) vector
    vectorized_matrix = input_matrix.reshape((input_size, 1))
    # Calculates values of partial derivatives with multiplication of the 
    # vectorized matrix by the specific Toeplitz matrix in a compressed
    # sparse row format
    matrices_product = toeplitz_sparse_matrix * vectorized_matrix
    # Reverts vectorized matrix of partial derivatives to a shape
    # of the input matrix
    p_derivative = matrices_product.reshape(output_shape)

    return p_derivative


def gaussian_weight(
        grad: np.ndarray,
        size: int,
        sigma: Union[int, float],
        epsilon: float
        ) -> np.ndarray:
    # """Initializes weight matrix according to the third wieght strategy of the 
    # original LIME paper.

    # Returns the shape-(M, N) array of weight values.

    # ## Args:
    #     grad (numpy.ndarray) : A shape-(M, N) array of partial gradient values.

    #     size (int) : An odd value which charactarizes the size of a Gaussian 
    #     kernel.

    #     sigma (int or float) : A standard deviation value of a Gaussian kernel.

    #     epsilon (float) : A small value which prevents division by zero 
    #     occurrences.

    # ## Returns:
    #     weights (numpy.ndarray) : A shape-(M, N) array of weights.
    # """

    radius=int((size-1)/2)
    denominator = epsilon + gaussian_filter(np.abs(grad), sigma, radius=radius, mode='constant')
    weights = gaussian_filter(1 / denominator, sigma, radius=radius, mode='constant')

    return weights


def initialize_weights(
        ill_map: np.ndarray,
        strategy_n: int,
        epsilon: float = 0.001
        ) -> np.ndarray:
    # """Initializes weight matrices according to a chosen strategy of 
    # the original LIME paper. Then updates and vectorizes these weight matrices 
    # preparing them to be used for calculation of a new illumination map. 

    # Returns the shape-(M, N) arrays of weight values with regard to horizontal 
    # and vertical directions.

    # """

    # Initializes weight matrices according to a chosen strategy
    if strategy_n == 1:
        print('Weight generation strategy: 1')
        weights = np.ones(ill_map.shape)
        weights_x = weights
        weights_y = weights
    elif strategy_n == 2:
        print('Weight generation strategy: 2')
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = 1 / (np.abs(grad_t_x) + epsilon)
        weights_y = 1 / (np.abs(grad_t_y) + epsilon)
    else:
        sigma = 2
        size = 15
        print('Weight generation strategy: 3')
        print(f'Strategy parameters: sigma = {sigma}, kernel size = {size}')
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = gaussian_weight(grad_t_x, size, sigma, epsilon)
        weights_y = gaussian_weight(grad_t_y, size, sigma, epsilon)
    # Modifies and transforms weight matrices in a vector form
    modified_w_x = weights_x / (np.abs(grad_t_x) + epsilon)
    modified_w_y = weights_y / (np.abs(grad_t_y) + epsilon)
    flat_w_x = modified_w_x.flatten()
    flat_w_y = modified_w_y.flatten()

    return flat_w_x, flat_w_y


def update_illumination_map(
        ill_map: np.ndarray,
        weight_strategy: int = 3
        ) -> np.ndarray:
  

    # Vectorizes initial illumination map
    vectorized_t = ill_map.reshape((ill_map.size, 1))
    epsilon = 0.001
    alpha = 0.15
    # Generates Toeplitz matrices of for computation of a forward difference 
    # in both horizontal and vertical directions
    row_sparsed, row2_sparsed = d_sparse_matrices(ill_map)
    # Initializes vectorized weight matrices according to a chosen strategy
    flatten_wiegths_x, flatten_wiegths_y = initialize_weights(
       ill_map, weight_strategy, epsilon)
    # Constructs diagonal matrices from vectorized weights
    diag_weights_x = diags(flatten_wiegths_x)
    diag_weights_y = diags(flatten_wiegths_y)
    # Updates the illumination map by solving the equation (19) of 
    # the original LIME paper
    x_term = row_sparsed.transpose() * diag_weights_x * row_sparsed
    y_term = row2_sparsed.transpose() * diag_weights_y * row2_sparsed
    identity = diags(np.ones(x_term.shape[0]))
    matrix = identity + alpha * (x_term + y_term)
    updated_t = spsolve(csr_matrix(matrix), vectorized_t)
    print('Solved:', type(updated_t), '\n')

    return updated_t.reshape(ill_map.shape)





