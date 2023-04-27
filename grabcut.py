import time

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

global g, beta, row, col, number_of_existing_edges, prev_energy, prev_diff

# k = infinity
K = 1e9


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    global prev_energy , prev_diff
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
        print("\nIteration:_______________ ", i)
        # Update GMM
        GMMUPDATE = time.time()
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        GMMUEND = time.time()
        print("GMM Update Time:_________", GMMUEND - GMMUPDATE)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        MASKUPDATE = time.time()
        print("MinCut Time :____________", MASKUPDATE - GMMUEND)
        mask = update_mask(mincut_sets, mask)
        MASKUEND = time.time()
        print("Mask Update Time:________", MASKUEND - MASKUPDATE)
        if check_convergence(energy):
            break
        CHECKCONV = time.time()
        print("Check Convergence Time:__", CHECKCONV - MASKUEND)
        print("energy:_________________ ", energy)
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    global g, beta, row, col, prev_energy, prev_diff
    debug = False
    # init
    row, col = img.shape[:2]
    if not debug:
        d_adjacent, d_below, diag1_n_link, diag2_n_link = calc_beta_and_n_link(img)
        weights = np.concatenate((d_adjacent, d_below, diag1_n_link, diag2_n_link))
        init_graph(weights)

    prev_energy = 0
    prev_diff = 0

    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    # Create an empty GMM for the foreground and background
    fgGMM = gmm_init()
    bgGMM = gmm_init()

    # Use KMeans to cluster the pixels into n_components clusters for each of foreground and background
    fg_kmeans = KMeans(n_clusters=n_components, random_state=0, n_init=10).fit(fg_pixels)
    bg_kmeans = KMeans(n_clusters=n_components, random_state=0, n_init=10).fit(bg_pixels)

    # Fill the GMM with the KMeans results
    gmm_fill(fgGMM, fg_kmeans, fg_pixels)

    gmm_fill(bgGMM, bg_kmeans, bg_pixels)

    return fgGMM, bgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    # Use KMeans to cluster the pixels into n_components clusters for each of foreground and background
    fg_kmeans = KMeans(n_clusters=n_components, random_state=0, n_init=10).fit(fg_pixels)
    bg_kmeans = KMeans(n_clusters=n_components, random_state=0, n_init=10).fit(bg_pixels)

    # Fill the GMM with the KMeans results
    gmm_fill(fgGMM, fg_kmeans, fg_pixels)
    gmm_fill(bgGMM, bg_kmeans, bg_pixels)

    return bgGMM, fgGMM


def add_t_links(t_s2, t_t2, mask):
    global g, number_of_existing_edges

    unknown, background = map_mask_to_img(mask)

    g.add_edges(zip(['s'] * len(unknown), unknown))
    g.add_edges(zip(['t'] * len(unknown), unknown))

    g.add_edges(zip(['s'] * len(background), background))
    g.add_edges(zip(['t'] * len(background), background))

    s_background = K * np.ones(len(background))
    t_background = np.zeros(len(background))

    t = np.concatenate((t_s2, t_t2, s_background, t_background))
    g.es[number_of_existing_edges:][WEIGHT] = t


# TODO: finish this function
def calculate_mincut(img, mask, bgGMM, fgGMM):
    global g
    t_link_s, t_link_t = t_link(img, mask, bgGMM, fgGMM)

    add_t_links(t_link_s, t_link_t, mask)

    min_cut = g.st_mincut(source='s', target='t', capacity=WEIGHT)

    energy = min_cut.value
    min_cut = min_cut.partition
    # rename the min_cut sets using the vertex_name function
    return min_cut, energy


def map_mask_to_img(mask):
    front = np.transpose(np.nonzero(mask))
    back = np.transpose(np.nonzero(3 - mask) or np.nonzero(1 - mask))
    front_pos = [vertex_name(front[i][0], front[i][1]) for i in range(front.shape[0])]
    back_pos = [vertex_name(back[i][0], back[i][1]) for i in range(back.shape[0])]
    return front_pos, back_pos


def update_mask(mincut_sets, mask):
    global row, col
    # get the foreground and background pixels from the mincut sets
    foreground = mincut_sets[1][1:]

    # create a mask with the same size as the image
    mask = np.zeros((row, col)).flatten()

    mask[foreground] = GC_PR_FGD

    mask = mask.reshape((row, col))

    return mask


def check_convergence(energy):
    global prev_energy, prev_diff
    curr_diff = abs(energy - prev_energy)
    if curr_diff - prev_diff < 0.01:
        convergence = True
        return convergence
    prev_diff = curr_diff
    prev_energy = energy
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def calc_beta_and_n_link(img_):
    global beta, row, col
    img = img_.astype(np.int16)
    # Calculate the differences between adjacent pixels in the image
    d_adjacent = np.diff(img, axis=1).reshape(-1, 3)  # I[i,j +1] - I[i,j]
    d_adjacent = d_adjacent.astype(np.int16)
    d_below = np.diff(img, axis=0).reshape(-1, 3)  # I[i+1,j] - I[i,j]
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
    neighbor_count = d_adjacent.shape[0] + d_below.shape[0] + diag1.shape[0] + diag2.shape[0]
    # Calculate the beta parameter
    beta = 1 / (2 * (sum_m / neighbor_count))

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


def add_n_links_edges(weights):
    global g, row, col, number_of_existing_edges
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
    number_of_existing_edges = len(g.get_edgelist())


def t_link(img, mask, bgGMM, fgGMM):
    img_masked = img[mask > 0]
    t_link_source, t_link_target = t_link_calc(img_masked, bgGMM, fgGMM)
    return -np.log(t_link_source), -np.log(t_link_target)


def t_link_calc(img_masked, bgGMM, fgGMM):
    # Calculate the numerator and denominator for the t-link equations
    diff_back = img_masked[:, np.newaxis] - bgGMM['means'][np.newaxis]
    diff_fore = img_masked[:, np.newaxis] - fgGMM['means'][np.newaxis]

    covs_back_inv = bgGMM['covs']
    covs_fore_inv = fgGMM['covs']

    left_factor_back = bgGMM['weights'] * bgGMM['dets']
    left_factor_fore = fgGMM['weights'] * fgGMM['dets']

    ut_a_u_back = np.array(list(
        (diff_back[i][j] @ covs_back_inv[j]) @ diff_back[i][j] for i in range(diff_back.shape[0]) for j in
        range(5))).reshape(diff_back.shape[0], 5)
    ut_a_u_fore = np.array(list(
        (diff_fore[i][j] @ covs_fore_inv[j]) @ diff_fore[i][j] for i in range(diff_fore.shape[0]) for j in
        range(5))).reshape(diff_fore.shape[0], 5)

    # Calculate the t-link source and target values
    t_link_source = np.sum(np.multiply(left_factor_back, np.exp(-0.5 * ut_a_u_back)), axis=1)
    t_link_target = np.sum(np.multiply(left_factor_fore, np.exp(-0.5 * ut_a_u_fore)), axis=1)

    return t_link_source, t_link_target


# TODO: maybe vectorize the function
def init_graph(weights):
    global g
    g = ig.Graph()
    add_nodes()
    add_n_links_edges(weights)


def add_nodes():
    global g, row, col
    # add nods to graph
    g.add_vertex('s')
    g.add_vertex('t')
    for i in range(row):
        for j in range(col):
            vertex_id = vertex_name(i, j)
            g.add_vertex(vertex_id)
    # add source and sink


def gmm_fill(GMM, kmeans, pixels):
    GMM['means'] = kmeans.cluster_centers_
    GMM['covs'] = np.array([np.cov(pixels[kmeans.labels_ == i].T) for i in range(n_components)])
    GMM['dets'] = np.array([np.linalg.det(GMM['covs'][i]) for i in range(n_components)])
    GMM['dets'] = 1 / np.sqrt(GMM['dets'])
    GMM['covs'] += np.eye(3) * 1e-6
    GMM['covs'] = np.linalg.inv(GMM['covs'])


def gmm_init():
    GMM = {'weights': np.full(n_components, 1 / n_components),
           'means': np.zeros((n_components, 3)),
           'covs': np.zeros((n_components, 3, 3)),
           'dets': np.zeros(n_components)}
    return GMM


def name_to_vertex(name):
    global col

    return int(name) // col, int(name) % col


def vertex_name(i, j):
    global col
    # return str(i) + ',' + str(j)
    return i * col + j


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
