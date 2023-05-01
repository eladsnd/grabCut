import argparse
import cv2
from scipy.sparse.linalg import spsolve
import numpy as np
from scipy.sparse import diags


# def create_laplacian_matrix(width, height):
#     N = width * height
#
#     # Diagonal entries
#     d0 = np.ones(N) * -4
#     d1 = np.ones(N - 1)
#     d2 = np.ones(N - width)
#
#     # Create diagonals of the matrix
#     diags_list = [d0, d1, d1, d2, d2]
#     diags_pos = [0, -1, 1, -width, width]
#
#     # Create sparse matrix
#     laplacian = diags(diags_list, diags_pos, shape=(N, N))
#
#     # Set boundary conditions
#     laplacian = laplacian.tolil()
#     for i in range(width):
#         laplacian[i, i] = -3
#         laplacian[N - i - 1, N - i - 1] = -3
#     for i in range(height):
#         laplacian[i * width, i * width] = -3
#         laplacian[(i + 1) * width - 1, (i + 1) * width - 1] = -3
#
#     # Convert back to CSR format
#     laplacian = laplacian.tocsr()
#
#     return laplacian

def create_laplacian_matrix(width, height):
    N = width * height
    d_main = np.ones(N) * -4
    d_off = np.ones(N - 1)
    d_off[::width] = 0
    d_off[-width::] = 0
    d_neigh = np.ones(N - width)
    diagonals = [d_main, d_off, d_off, d_neigh, d_neigh]
    offsets = [0, -1, 1, -width, width]
    laplacian = diags(diagonals, offsets, shape=(N, N), format='csr')
    laplacian[N-1, N-1] = 1
    return laplacian

# def create_boundary_vector(im_src, im_tgt, im_mask, center, laplacian):
#     # Create mask for boundary pixels
#     mask_boundary = np.logical_and(im_mask, 1 - cv2.erode(im_mask, None))
#
#     # Create boundary indices
#     boundary_indices = np.arange(im_src.size)[mask_boundary.flatten()]
#
#     # Create boundary vector
#     boundary_vector = np.zeros(im_src.size)
#     boundary_vector[boundary_indices] = im_tgt.flatten()[boundary_indices] - im_src.flatten()[boundary_indices]
#
#     # Subtract the laplacian of the source image from the boundary vector
#     boundary_vector[boundary_indices] -= laplacian.dot(im_src.flatten())[boundary_indices]
#
#     # Set the center pixel of the boundary vector to the corresponding pixel in the source image
#     boundary_vector[center[0]] = im_src.flatten()[center[0]]
#     boundary_vector[center[1]] = im_src.flatten()[center[1]]
#
#     return boundary_vector.flatten().reshape(-1, 1)


def create_boundary_vector(im_src, im_tgt, im_mask, center, laplacian, channel):
    # Create mask for boundary pixels
    mask_boundary = np.logical_and(im_mask, 1 - cv2.erode(im_mask, None))

    # Create boundary indices
    boundary_indices = np.arange(im_src.size)[mask_boundary.flatten()]

    # Create boundary vector
    boundary_vector = np.zeros(im_src.size)
    boundary_vector[boundary_indices] = im_tgt.flatten()[boundary_indices + channel * im_src.size] - im_src.flatten()[
        boundary_indices + channel * im_src.size]

    # Subtract the laplacian of the source image from the boundary vector
    boundary_vector[boundary_indices] -= laplacian.dot(im_src[:, :, channel].flatten())[boundary_indices]

    # Set the center pixel of the boundary vector to the corresponding pixel in the source image
    boundary_vector[center] = im_src.flatten()[center + channel * im_src.size]

    return boundary_vector.reshape(-1, 1)

def reshape_to_target(im_mask, im_src, im_tgt):
    # The mask and source images are first padded so that they are the same size as the target image
    im_mask = np.pad(im_mask, ((0, im_tgt.shape[0] - im_mask.shape[0]), (0, im_tgt.shape[1] - im_mask.shape[1])),
                     'constant', constant_values=0)
    im_src = np.pad(im_src, ((0, im_tgt.shape[0] - im_src.shape[0]), (0, im_tgt.shape[1] - im_src.shape[1]), (0, 0)),
                    'constant', constant_values=0)
    return im_src, im_mask

def poisson_blend(im_src, im_tgt, im_mask, center):
    """
    Perform Poisson blending between a source and target image using a binary mask.

    Args:
        im_src (ndarray): The source image to be blended.
        im_tgt (ndarray): The target image onto which the source will be blended.
        im_mask (ndarray): A binary mask indicating the region of the source to be blended.
        center (tuple): The (x,y) coordinates of the center of the source region.

    Returns:
        ndarray: The blended image.
    """

    im_src, im_mask = reshape_to_target(im_mask, im_src, im_tgt)

    # Split image into channels
    im_src_channels = np.rollaxis(im_src, 2)
    im_tgt_channels = np.rollaxis(im_tgt, 2)

    # Initialize blended image
    im_blend_channels = np.zeros_like(im_src_channels)

    # Create Laplacian matrix
    laplacian = create_laplacian_matrix(im_src.shape[:2][0], im_src.shape[:2][1])

    # Iterate over channels
    for i in range(im_src_channels.shape[0]):
        # Create boundary vector
        b = create_boundary_vector(im_src_channels[i], im_tgt_channels[i], im_mask, center, laplacian, im_src_channels[i])

        # Solve the system using sparse matrices
        x = spsolve(laplacian, b)

        # Reshape solution into image dimensions
        im_blend_channels[i] = x.reshape(im_src.shape[:2])

    # Merge channels and return blended image
    return np.rollaxis(im_blend_channels, 0, 3)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))
    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()