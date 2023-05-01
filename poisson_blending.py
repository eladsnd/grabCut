import cv2
import numpy as np
import argparse
from scipy import sparse, ndimage
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags, eye, lil_matrix
from scipy.sparse.linalg import spsolve

# def poisson_blend(im_src, im_tgt, im_mask, center):
#     # TODO: Implement Poisson blending of the source image onto the target ROI
#
#     im_blend = im_tgt
#     return im_blend


def compute_laplacian(img):
    # Define Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Compute Laplacian using convolution with kernel
    laplacian = ndimage.convolve(img.astype(np.float32), kernel)

    return laplacian

def create_laplacian_matrix(width, height):
    n = width * height
    d = np.ones(n)
    d[width-1::width] = 0
    d[-width:] = 0
    d[0:width] = 0
    d[0::width] = 0

    diags = np.array([0, 1, -1, width, -width])
    laplacian = spdiags([-4*d, d, d, d, d], diags, n, n)
    return laplacian

def create_boundary_vector(channel_src, channel_tgt, mask, center, laplacian):
    """
    Creates the boundary vector for a single color channel.

    Args:
        channel_src: numpy array representing the source image channel.
        channel_tgt: numpy array representing the target image channel.
        mask: numpy array representing the binary mask of the source region.
        center: tuple representing the center of the source region.
        laplacian: sparse matrix representing the Laplacian operator.

    Returns:
        numpy array representing the boundary vector.
    """
    h, w = channel_src.shape[:2]
    b = np.zeros((h, w))

    # find the indices of the source region in the mask
    indices = np.nonzero(mask)

    # calculate the Laplacian of the target region
    laplacian_tgt = laplacian.dot(channel_tgt.flatten())

    # calculate the boundary vector
    for x, y in zip(indices[0], indices[1]):
        if y == 0 or y == w-1 or x == 0 or x == h-1:
            # pixel is on the border
            b[x, y] = channel_src[x, y]
        else:
            # pixel is not on the border, calculate the Laplacian of the source region
            laplacian_src = 4 * channel_src[x, y] - channel_src[x-1, y] - channel_src[x+1, y] \
                            - channel_src[x, y-1] - channel_src[x, y+1]

            # calculate the boundary value
            b[x, y] = laplacian_tgt[x*w + y] - laplacian_src

    # set the center pixel value
    # center_x, center_y = center
    # b[center_y, center_x] = channel_src[center_y, center_x]

    # return the boundary vector as a 1D array
    return b.flatten()



def poisson_blend(im_src, im_tgt, im_mask, center):
    """
    Perform Poisson blending of the source image onto the target image
    within the given mask region, centered at the given point.
    """

    # Get image dimensions
    M, N, channels = im_tgt.shape

    # Create sparse Laplacian operator matrix
    laplacian = create_laplacian_matrix(M, N)

    # Create sparse identity matrix
    identity = eye(laplacian.shape[0], format='csr')

    # Create sparse mask matrix
    mask_indices = np.where(im_mask.flatten())
    mask_matrix = lil_matrix((M*N, M*N))
    # mask_matrix[mask_indices[0], mask_indices[0]] = 1

    A = mask_matrix.dot(identity - laplacian)

    # Compute blended image for each color channel
    blended_channels = []
    for channel in range(im_src.shape[2]):
        # Create boundary vector b
        b = create_boundary_vector(im_src[:, :, channel], im_tgt[:, :, channel], im_mask, center, laplacian)

        # Solve the Poisson equation to obtain the blended image for this channel
        x = spsolve(A, b)
        x = np.clip(x, 0, 1)

        blended_channels.append(x.reshape(im_src.shape[:2]))

    # Merge blended channels into RGB image
    blended = np.stack(blended_channels, axis=2)

    return blended

def poisson_blending(im_src, im_tgt, im_mask, center):
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
    lap_target = compute_laplacian(im_tgt)

    # Construct sparse matrix A for Poisson equation
    A = diags([mask], [0], format="lil") + lap_target

    # Compute Laplacian of source patch
    lap_source = lap_target * src_patch.reshape(-1, 3)

    # Solve Poisson equation to obtain blended patch
    blended_patch_flat = spsolve(A, lap_source).reshape(-1, 3)

    # Clip blended patch to [0, 255]
    blended_patch = np.clip(blended_patch_flat, 0, 255).astype(np.uint8)

    # Replace target patch with blended patch
    im_tgt[src_y : src_y + M, src_x : src_x + N, :] = blended_patch

    im_blend = im_tgt
    return im_blend

def poisson_blending_2(im_src, im_tgt, im_mask, center):
    # Compute Laplacian operator for each color channel
    laplacian_matrix = create_laplacian_matrix(im_mask)

    # Solve Poisson equation for each color channel
    im_blend = np.zeros_like(im_src)
    for i in range(im_src.shape[2]):
        A = laplacian_matrix
        b = create_boundary_vector(im_src[:, :, i], im_tgt[:, :, i], im_mask, center)
        x = sparse.linalg.spsolve(A, b)
        im_blend[:, :, i] = x.reshape(im_src.shape[:2])

    return im_blend

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