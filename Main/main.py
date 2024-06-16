import cv2
import numpy as np
import Modification_functions as mod
import bool_imag_check as check
import gamma_correc as gc
import os as os
from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d
import denoising_bm3d as db3d


folder = './test/low/'
# image = 'istockphoto-1325587373-612x612.jpg'
# img_path = os.path.join(folder, image)
# image_read = cv2.imread(img_path, cv2.IMREAD_COLOR)
# cv2.imshow('Image', image_read)
# cv2.waitKey(1000)


output_dir = './test/predicted/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
weig_strategy = 3
gamma_value = 0.4
standard_dev = 0.03

onlyimages = [f for f in listdir(folder) if check.bool_image(f)]
print(len(onlyimages))
for image in onlyimages:
    img_path = folder + image
    img_read = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(f'Image: {img_path}')
    # print(f'Image: {img_path}, resolution: {image_read.shape}')
    
    # Convert the image to RGB format and normalize pixel values to [0, 1] range
    img_rgb = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB) / 255

# Calculate the illumination map by finding the maximum value across the RGB channels
    ill_map = np.max(img_rgb, axis=-1)

# Update the illumination map using the specified weight strategy
    upd_ill_map = mod.update_illumination_map(ill_map, weig_strategy)

# Apply gamma correction to the updated illumination map
    cor_ill_map = gc.gamma_correct(np.abs(upd_ill_map), gamma_value)

# Expand the corrected illumination map dimensions to match the image dimensions
    cor_ill_map = cor_ill_map[..., np.newaxis]

# Correct the image illumination by dividing the image by the corrected illumination map
    new_img = img_rgb / cor_ill_map

# Clip the image values to [0, 1] range and convert to float32 type
    new_img = np.clip(new_img, 0, 1).astype("float32")

# Denoise the image using BM3D YUV denoising method with the specified standard deviation
    denoised_img = db3d.denoising_bm3d(new_img, cor_ill_map, standard_dev)
    
    plt.figure(figsize = (15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img)
    denoised_image_bgr = cv2.cvtColor(denoised_img * 255, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_dir + 'modified_' + image, denoised_image_bgr)

plt.show()