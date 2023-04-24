import numpy as np
import cv2
import argparse
import igraph as ig
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


def add_n_links(g, pixels, beta):
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            vertex_id = i * pixels.shape[1] + j
            # add n-links to graph for each neighboring pixel down and right diagonally down and right diagonally
            # down and left
            if i + 1 < pixels.shape[0]:  # [i+1, j]
                weight = n_link_calc()
                g.add_edge(vertex_id,
                           vertex_id + pixels.shape[1],
                           weight=weight)
            if j + 1 < pixels.shape[1]:  # [i, j+1]
                g.add_edge(vertex_id,
                           vertex_id + 1,
                           weight=n_link_calc(pixels[i, j], pixels[i, j + 1], beta))
            if i + 1 < pixels.shape[0] and j + 1 < pixels.shape[1]:  # [i+1, j+1]
                g.add_edge(vertex_id,
                           vertex_id + pixels.shape[1] + 1,
                           weight=n_link_calc(pixels[i, j], pixels[i + 1, j + 1], beta))
            if i + 1 < pixels.shape[0] and j - 1 >= 0:  # [i+1, j-1]
                g.add_edge(vertex_id,
                           vertex_id + pixels.shape[1] - 1,
                           weight=n_link_calc(pixels[i, j], pixels[i + 1, j - 1], beta))
    return g

def n_link_calc(img, i1, j1, i2, j2, beta):
    """
    n(x,y) = 50/dist(I(x),I(y)) * exp(-beta * ||I(x)-I(y)||^2)
    """
    dist = distance_between_pixels(img[i1, j1] - img[i2, j2])
    return 50 / dist * np.exp(-beta * dist ** 2)


def initalize_graph(pixels, beta):
    g = ig.Graph()
    add_nods(g, pixels)
    add_n_links(g, pixels, beta)


def add_nods(g, pixels):
    # add nods to graph
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            vertex_id = i * pixels.shape[1] + j
            g.add_vertex(vertex_id)
    # add source and sink
    g.add_vertex('s')
    g.add_vertex('t')


def initalize_GMMs(img, mask):
    beta = calc_beta(img)
    initalize_graph(img, beta)
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

def t_link(img, mask, bgGMM, fgGMM):
    for pixel in img[mask > 0]:
        t_link_calc(img, pixel,bgGMM,fgGMM)

def t_link_calc(img, pixel, bgGMM, fgGMM):
    sum_back = 0
    sum_fore = 0

    for i in range(5):
        sum_back += calc_product(img, pixel, i, bgGMM)
        sum_fore += calc_product(img, pixel, i, fgGMM)

    t_link_source = -np.log(sum_back)
    t_link_target = -np.log(sum_fore)
    return t_link_source, t_link_target

def calc_product(img, pixel, i, GMM):
    left_factor = GMM['weights'][i] * (1/np.sqrt(GMM['dets'][i]))
    right_factor = np.transpose((img[pixel]-GMM['means'][i])) * np.linalg.inv(GMM['covs'][i]) * (img[pixel]-GMM['means'][i])
    right_factor = np.exp(0.5 * right_factor)

    product = left_factor * right_factor

    return product


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
    diag1 = list((img[i + 1, j + 1] - img[i, j] for i in range(img.shape[0] - 1) for j in range(img.shape[1] - 1)))
    diag2 = list((img[i + 1, j - 1] - img[i, j] for i in range(img.shape[0] - 1) for j in range(1, img.shape[1])))
    diag1 = np.array(diag1)
    diag2 = np.array(diag2)

    # Calculate the sum of squared differences
    sum_m = np.sum(dx ** 2) + np.sum(dy ** 2) + np.sum(diag1 ** 2) + np.sum(diag2 ** 2)
    count = dx.shape[0] + dy.shape[0] + diag1.shape[0] + diag2.shape[0]
    # Calculate the beta parameter
    beta = 1 / (2 * (sum_m / count))
    return beta


def distance_between_pixels(pixel1, pixel2):
    return np.linalg.norm(pixel1 - pixel2)


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
