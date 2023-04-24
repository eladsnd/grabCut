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

WEIGHT = 'weight'

global g, beta


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # init the inner square to Foreground
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
    global g, beta
    d_adjacent, d_below, diag1_n_link, diag2_n_link = calc_beta_and_n_link(img)
    weights = np.concatenate((d_adjacent, d_below, diag1_n_link, diag2_n_link))

    init_graph(img, weights)

    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    # Create an empty GMM for the foreground and background
    fgGMM = gmm_init()
    bgGMM = gmm_init()

    # Use KMeans to cluster the pixels into n_components clusters for each of foreground and background
    fg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(fg_pixels)
    bg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(bg_pixels)

    # Fill the GMM with the KMeans results
    gmm_fill(fgGMM, fg_kmeans, fg_pixels)

    gmm_fill(bgGMM, bg_kmeans, bg_pixels)

    return fgGMM, bgGMM


def gmm_fill(GMM, kmeans, pixels):
    GMM['means'] = kmeans.cluster_centers_
    GMM['coves'] = np.array([np.cov(pixels[kmeans.labels_ == i].T) for i in range(n_components)])
    GMM['dets'] = np.array([np.linalg.det(GMM['coves'][i]) for i in range(n_components)])
    for i in range(n_components):
        GMM['coves'][i] = np.linalg.inv(GMM['coves'][i])


def gmm_init():
    GMM = {'weights': np.full(n_components, 1 / n_components),
           'means': np.zeros((n_components, 3)),
           'coves': np.zeros((n_components, 3, 3)),
           'dets': np.zeros(n_components)}
    return GMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    # Use KMeans to cluster the pixels into n_components clusters for each of foreground and background
    fg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(fg_pixels)
    bg_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(bg_pixels)

    # Fill the GMM with the KMeans results
    gmm_fill(fgGMM, fg_kmeans, fg_pixels)
    gmm_fill(bgGMM, bg_kmeans, bg_pixels)

    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    global g
    min_cut, cut_partition = g.st_mincut(source='s', target='t', capacity=WEIGHT)

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


def calc_beta_and_n_link(img):
    global beta
    row = img.shape[0]
    col = img.shape[1]
    # Calculate the differences between adjacent pixels in the image
    d_adjacent = np.diff(img, axis=1).reshape(-1, 3)  # I[i,j] - I[i,j+1]
    d_below = np.diff(img, axis=0).reshape(-1, 3)  # I[i,j] - I[i+1,j]
    diag1 = np.array(
        list((img[i + 1, j + 1] - img[i, j] for i in range(row - 1) for j in range(col - 1))))
    diag2 = np.array(
        list((img[i + 1, j - 1] - img[i, j] for i in range(row - 1) for j in range(1, col))))

    dx_sum_dist_square = sum_distance_square(d_adjacent)
    dy_sum_dist_square = sum_distance_square(d_below)
    diag1_sum_dist_square = sum_distance_square(diag1)
    diag2_sum_dist_square = sum_distance_square(diag2)

    # Calculate the sum of the squared differences
    sum_m = sum(diag1_sum_dist_square) + sum(diag2_sum_dist_square) + sum(dx_sum_dist_square) + sum(dy_sum_dist_square)
    nighbor_count = d_adjacent.shape[0] + d_below.shape[0] + diag1.shape[0] + diag2.shape[0]
    # Calculate the beta parameter
    beta = 1 / (2 * (sum_m / nighbor_count))

    # Calculate the n-link weights
    dx_n_link = n_link_calc(dx_sum_dist_square)
    dy_n_link = n_link_calc(dy_sum_dist_square)
    diag1_n_link = n_link_calc(diag1_sum_dist_square)
    diag2_n_link = n_link_calc(diag2_sum_dist_square)

    return dx_n_link, dy_n_link, diag1_n_link, diag2_n_link


def sum_distance_square(rgb_vector):
    rgb_vector_sum_square = np.array(list((np.sum(np.multiply(rgb_vector[i], rgb_vector[i]))
                                           for i in range(rgb_vector.shape[0]))))
    return rgb_vector_sum_square


def n_link_calc(sum_dist_square):
    global beta
    return 50 * np.multiply(np.exp(-beta * sum_dist_square),
                            np.where((sum_dist_square > 0), (1 / (np.sqrt(sum_dist_square))), 0))


def add_n_links_edges(img, weights):
    global g
    row = img.shape[0]
    col = img.shape[1]

    # Add the n-link edges to the graph
    dx_edges = list((vertex_name(i, j), vertex_name(i, j + 1))
                    for i in range(row) for j in range(col - 1))
    dy_edges = list((vertex_name(i, j), vertex_name(i + 1, j))
                    for i in range(row - 1) for j in range(col))
    diag1_edges = list((vertex_name(i, j), vertex_name(i + 1, j + 1))
                       for i in range(row - 1) for j in range(col - 1))
    diag2_edges = list((vertex_name(i, j), vertex_name(i + 1, j - 1))
                       for i in range(row - 1) for j in range(1, col))

    # concatenate all weights
    edges = np.concatenate((dx_edges, dy_edges, diag1_edges, diag2_edges))

    g.add_edges(edges)
    g.es[WEIGHT] = weights

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

def init_graph(img, weights):
    global g
    g = ig.Graph()
    add_nodes(img)
    add_n_links_edges(img, weights)


def add_nodes(img):
    global g
    # add nods to graph
    g.add_vertex('s')
    g.add_vertex('t')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            vertex_id = vertex_name(i, j)
            g.add_vertex(vertex_id)
    # add source and sink


def vertex_name(i, j):
    return "(" + str(i) + ',' + str(j) + ")"


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
