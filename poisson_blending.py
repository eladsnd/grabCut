import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse
from scipy.sparse import csr_matrix, diags, kron
from scipy.sparse.linalg import spsolve


# def poisson_blend(im_src, im_tgt, im_mask, center):
#     # TODO: Implement Poisson blending of the source image onto the target ROI
#
#     im_blend = im_tgt
#     return im_blend


def laplacian_operator(n):
    """
    Computes the 2D Laplacian operator for an n x n image.
    """
    lap = diags([-4, 1, 1], [0, -1, 1], shape=(n, n)).tolil()
    lap[0, 1] = 2
    lap[-1, -2] = 2
    return csr_matrix(kron(lap, diags([1], [0], shape=(n, n))))


def poisson_blend(im_src, im_tgt, im_mask, center):
    """
    Performs Poisson blending of a source image onto a target image using a mask.

    Args:
        im_src (numpy.ndarray): Source image (M x N x 3)
        im_tgt (numpy.ndarray): Target image (M x N x 3)
        im_mask (numpy.ndarray): Mask image (M x N)
        center (tuple): Center point (x, y) for placing the source image in the target image.

    Returns:
        numpy.ndarray: Blended image (M x N x 3)
    """
    # Convert images to float32 for precision
    im_src = im_src.astype(np.float32)
    im_tgt = im_tgt.astype(np.float32)

    # Get image dimensions
    M, N, _ = im_tgt.shape

    # Extract source patch centered at target image
    src_x, src_y = center
    src_patch = im_src[src_y : src_y + M, src_x : src_x + N, :]

    # Flatten mask into a 1D array
    mask = im_mask.flatten()

    # Compute Laplacian operator for target image
    lap_target = laplacian_operator(M * N)

    # Construct sparse matrix A for Poisson equation
    A = diags([mask], [0], shape=(M * N, M * N)) + lap_target

    # Compute Laplacian of source patch
    lap_source = laplacian_operator(M * N) * src_patch.reshape(-1, 3)

    # Solve Poisson equation to obtain blended patch
    blended_patch_flat = spsolve(A, lap_source).reshape(-1, 3)

    # Clip blended patch to [0, 255]
    blended_patch = np.clip(blended_patch_flat, 0, 255).astype(np.uint8)

    # Replace target patch with blended patch
    im_tgt[src_y : src_y + M, src_x : src_x + N, :] = blended_patch

    im_blend = im_tgt
    return im_blend

# # Load source, target, and mask images
# im_src = cv2.imread('source.jpg')
# im_tgt = cv2.imread('target.jpg')
# im_mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Specify center point for placing source image in target image
# center = (100, 100)
#
# # Perform Poisson blending
# im_blended = poisson_blend(im_src, im_tgt, im_mask, center)
#
# # Save blended image
# cv2.imwrite('blended.jpg', im_blended)

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