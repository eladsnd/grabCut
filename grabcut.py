import numpy as np
import cv2
import argparse

from sklearn.cluster import KMeans

n_components = 5

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    beta = calc_beta(img)
    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    # Use KMeans to cluster the pixels into n_components clusters for each of foreground and background
    fg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(fg_pixels)
    bg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(bg_pixels)

    # Create an empty GMM for the foreground and background
    fgGMM = {
        'weights': np.zeros(n_components),
        'means': np.zeros((n_components, 3)),
        'covs': np.zeros((n_components, 3, 3)),
        'dets': np.zeros(n_components)
    }

    bgGMM = {
        'weights': np.zeros(n_components),
        'means': np.zeros((n_components, 3)),
        'covs': np.zeros((n_components, 3, 3)),
        'dets': np.zeros(n_components)}

    # Fill the GMM with the KMeans results
    fgGMM['weights'] = np.full(n_components, 1 / n_components)
    fgGMM['means'] = fg_kmeans.cluster_centers_
    fgGMM['covs'] = np.array([np.cov(fg_pixels[fg_kmeans.labels_ == i].T) for i in range(n_components)])
    fgGMM['dets'] = np.array([np.linalg.det(fgGMM['covs'][i]) for i in range(n_components)])
    for i in range(n_components):
        fgGMM['covs'][i] = np.linalg.inv(fgGMM['covs'][i])

    bgGMM['weights'] = np.full(n_components, 1 / n_components)
    bgGMM['means'] = bg_kmeans.cluster_centers_
    bgGMM['covs'] = np.array([np.cov(bg_pixels[bg_kmeans.labels_ == i].T) for i in range(n_components)])
    bgGMM['dets'] = np.array([np.linalg.det(bgGMM['covs'][i]) for i in range(n_components)])
    for i in range(n_components):
        bgGMM['covs'][i] = np.linalg.inv(bgGMM['covs'][i])

    return fgGMM, bgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    # Use KMeans to cluster the pixels into n_components clusters for each of foreground and background
    fg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(fg_pixels)
    bg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(bg_pixels)

    # Fill the GMM with the KMeans results
    # fgGMM['weights'] = np.full(n_components, 1 / n_components)
    fgGMM['means'] = fg_kmeans.cluster_centers_
    fgGMM['covs'] = np.array([np.cov(fg_pixels[fg_kmeans.labels_ == i].T) for i in range(n_components)])
    fgGMM['dets'] = np.array([np.linalg.det(fgGMM['covs'][i]) for i in range(n_components)])
    for i in range(n_components):
        fgGMM['covs'][i] = np.linalg.inv(fgGMM['covs'][i])

    # bgGMM['weights'] = np.full(n_components, 1 / n_components)
    bgGMM['means'] = bg_kmeans.cluster_centers_
    bgGMM['covs'] = np.array([np.cov(bg_pixels[bg_kmeans.labels_ == i].T) for i in range(n_components)])
    bgGMM['dets'] = np.array([np.linalg.det(bgGMM['covs'][i]) for i in range(n_components)])
    for i in range(n_components):
        bgGMM['covs'][i] = np.linalg.inv(bgGMM['covs'][i])
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    min_cut = [[], []]
    energy = 0
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def calc_beta(img):
    """
    Calculates the beta parameter for a given image.

    Args:
    img (numpy.ndarray): A 2D numpy array representing the image.

    Returns:
    float: The calculated beta value.
    """

    # Calculate the differences between adjacent pixels in the image
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    diag1 = list((img[i+1, j+1] - img[i, j] for i in range(img.shape[0]-1) for j in range(img.shape[1]-1)))
    diag1 = np.array(diag1)

    # Calculate the sum of squared differences
    sum_m = np.sum(dx ** 2) + np.sum(dy ** 2) + np.sum(diag1 ** 2)

    # Calculate the beta parameter
    beta = 1 / (2 * sum_m / ((img.shape[0] - 1) * (img.shape[1] - 1) * 3))

    return beta


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
