import math
import time

import numpy as np
import cv2
import argparse
import igraph as ig
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

n_components = 2

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

WEIGHT = 'weight'

global g, beta, row, col, number_of_existing_edges, prev_energy, K, number_of_edges_to_delete
FG = 'fg'
BG = 'bg'


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    global prev_energy
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # init the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD
    INIT = time.time()
    bgGMM, fgGMM, weights = initalize_GMMs(img, mask)
    INITEND = time.time()
    # print("Initialization Time:_____", INITEND - INIT)
    num_iters = 1000
    for i in range(num_iters):
        print("\nIteration:_______________ ", i)
        # Update GMM
        GMMUPDATE = time.time()
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        GMMUEND = time.time()
        # print("GMM Update Time:_________", GMMUEND - GMMUPDATE)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        MASKUPDATE = time.time()
        # print("MinCut Time :____________", MASKUPDATE - GMMUEND)
        mask = update_mask(mincut_sets, mask)
        MASKUEND = time.time()
        # print("Mask Update Time:________", MASKUEND - MASKUPDATE)
        if check_convergence(energy):
            break
        CHECKCONV = time.time()
        # print("Check Convergence Time:__", CHECKCONV - MASKUEND)
        print("energy:_________________ ", energy)
        # if i == 0:
        #     break
        # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    global g, beta, row, col, prev_energy, K, number_of_edges_to_delete
    # init global variables

    row, col = img.shape[:2]
    number_of_edges_to_delete = 0
    prev_energy = float('inf')
    prev_diff = 0

    # init graph with n_links
    d_adjacent, d_below, diag1_n_link, diag2_n_link = calc_beta_and_n_link(img)
    weights = np.concatenate((d_adjacent, d_below, diag1_n_link, diag2_n_link))
    init_graph(weights)
    K = 50000

    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    # Create an empty GMM for the foreground and background
    fgGMM = gmm_init()
    bgGMM = gmm_init()

    # Use KMeans to cluster the pixels into n_components clusters for each of foreground and background
    bg_kmeans = KMeans(n_clusters=n_components, random_state=0, n_init=10).fit(bg_pixels)
    fg_kmeans = KMeans(n_clusters=n_components, random_state=0, n_init=10).fit(fg_pixels)

    # Fill the GMM with the KMeans results
    gmm_fill_init(fgGMM, fg_kmeans, fg_pixels)
    gmm_fill_init(bgGMM, bg_kmeans, bg_pixels)

    return bgGMM, fgGMM, weights


def update_GMMs(img, mask, bgGMM, fgGMM):
    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    fg_components = np.zeros((fg_pixels.shape[0], fgGMM['n_components']))
    bg_components = np.zeros((bg_pixels.shape[0], bgGMM['n_components']))

    for i in range(fgGMM['n_components']):
        fg_components[:, i] = multivariate_normal.pdf(fg_pixels, fgGMM["means"][i], fgGMM["covs"][i],
                                                      allow_singular=True)
    for i in range(bgGMM['n_components']):
        bg_components[:, i] = multivariate_normal.pdf(bg_pixels, bgGMM["means"][i], bgGMM["covs"][i],
                                                      allow_singular=True)

    fg_new_indices = np.argmax(fg_components, axis=1)
    bg_new_indices = np.argmax(bg_components, axis=1)

    gmm_fill_update(fgGMM, fg_pixels, fg_new_indices)
    gmm_fill_update(bgGMM, bg_pixels, bg_new_indices)


    return bgGMM, fgGMM


def f_x(x, E, left_factor):
    x_E_x = np.sum(x * np.dot(x, E), axis=1)
    calc = left_factor * np.exp(-0.5 * x_E_x)
    return calc


def gmm_fill_update(GMM, pixels, new_indices):
    sum_weights = 0
    num_comp = 0
    for i in range(GMM['n_components']):
        gmm_pixels = pixels[np.where(new_indices == i)]
        if gmm_pixels.shape[0] > 1:
            GMM['means'][num_comp] = np.mean(gmm_pixels, axis=0)
            GMM['covs'][num_comp] = np.cov(gmm_pixels.T)
            GMM['covs'][num_comp] += np.eye(3) * 1e-6
            GMM['weights'][num_comp] = gmm_pixels.shape[0]
            while np.linalg.det(GMM['covs'][num_comp]) <= 0:
                GMM['covs'][num_comp] += np.eye(3) * 1e-6
            GMM['dets'][num_comp] = 1/np.linalg.det(GMM['covs'][num_comp])
            GMM['inv_covs'][num_comp] = np.linalg.inv(GMM['covs'][num_comp])
            sum_weights += GMM['weights'][num_comp]
            num_comp += 1
        else:
            GMM['weights'][i] = 0
            GMM['means'][i] = np.zeros(3)
            GMM['covs'][i] = np.eye(3)
            GMM['dets'][i] = 1
            GMM['inv_covs'][i] = np.eye(3)

    GMM['weights'] = GMM['weights'] / sum_weights
    GMM['n_components'] = num_comp
    # reshape_GMM(GMM)
    GMM['means'] = GMM['means'][:num_comp]
    GMM['covs'] = GMM['covs'][:num_comp]
    GMM['dets'] = GMM['dets'][:num_comp]
    GMM['inv_covs'] = GMM['inv_covs'][:num_comp]
    GMM['weights'] = GMM['weights'][:num_comp]


def print_all_gmms_data(bgGMM, fgGMM):
    print("~~BG GMM~~")
    print_all_gmm_data(bgGMM)
    print("~~FG GMM~~")
    print_all_gmm_data(fgGMM)


def print_all_gmm_data(gmm):
    for i in range(n_components):
        print(f"comp {i}")
        print(f"mean: {gmm['means'][i]}")
        print(f"cov: {gmm['dets'][i]}")
        print(f"weight: {gmm['weights'][i]}")


def add_t_links(bg_t_link, fg_t_link, mask):
    global g, number_of_existing_edges, number_of_edges_to_delete, K
    if number_of_edges_to_delete > 0:
        edges_to_delete = g.es[-number_of_edges_to_delete:]
        g.delete_edges(edges_to_delete)

    strong_fg_pixels_pos, fg_pixels_pos, bg_pixels_pos = map_mask_to_img(mask)
    # D(front) [for background] , D(back) [for foreground]
    g.add_edges(zip([BG] * len(fg_pixels_pos), fg_pixels_pos))
    g.add_edges(zip([FG] * len(fg_pixels_pos), fg_pixels_pos))

    # Strong background t-links
    g.add_edges(zip([BG] * len(bg_pixels_pos), bg_pixels_pos))
    g.add_edges(zip([FG] * len(bg_pixels_pos), bg_pixels_pos))

    # Strong foreground t-links
    g.add_edges(zip([BG] * len(strong_fg_pixels_pos), strong_fg_pixels_pos))
    g.add_edges(zip([FG] * len(strong_fg_pixels_pos), strong_fg_pixels_pos))

    bg_to_BG = K * np.ones(len(bg_pixels_pos))
    bg_to_FG = np.zeros(len(bg_pixels_pos))

    sfg_to_BG = np.zeros(len(strong_fg_pixels_pos))
    sfg_to_FG = K * np.ones(len(strong_fg_pixels_pos))

    t = np.concatenate((bg_t_link, fg_t_link, bg_to_BG, bg_to_FG, sfg_to_BG, sfg_to_FG))

    g.es[number_of_existing_edges:][WEIGHT] = t
    number_of_edges_to_delete = len(t)


# TODO: finish this function
def calculate_mincut(img, mask, bgGMM, fgGMM):
    D_fore, D_back = t_link(img, mask, bgGMM, fgGMM)

    add_t_links(D_back, D_fore, mask)

    min_cut = g.st_mincut(source=FG, target=BG, capacity=WEIGHT)

    energy = min_cut.value
    min_cut = min_cut.partition

    return min_cut, energy


def map_mask_to_img(mask):
    flat_mask = mask.flatten()
    fg_pos = np.where(flat_mask == GC_PR_FGD)[0]
    back_pos = np.where(flat_mask == GC_BGD)[0]
    sfg_pos = np.where(flat_mask == GC_FGD)[0]
    return sfg_pos, fg_pos, back_pos


def update_mask(mincut_sets, mask):
    global row, col

    foreground = mincut_sets[0]
    background = mincut_sets[1]

    # remove vertex s and t from the sets
    foreground.remove(g.vs.find(name=FG).index)
    background.remove(g.vs.find(name=BG).index)

    foreground = np.array(foreground)
    background = np.array(background)

    mask = mask.flatten()

    strong_foreground = np.where(mask == GC_FGD)[0]

    mask[foreground] = GC_PR_FGD
    mask[background] = GC_BGD
    mask[strong_foreground] = GC_FGD

    mask = mask.reshape((row, col))
    # create a mask with the same size as the image
    return mask


def check_convergence(energy):
    global prev_energy
    diff = abs(energy - prev_energy)
    print("curr_diff: ", diff)
    if diff < 1000:
        convergence = True
        return convergence
    prev_energy = energy
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # (the number of pixels that are correctly labeled divided by the total number of pixels in the image)
    correctly_labeled = np.sum(predicted_mask == gt_mask)
    total_pixels = gt_mask.shape[0] * gt_mask.shape[1]
    metric = correctly_labeled / total_pixels
    # Jaccard similarity (the intersection over the union of your predicted foreground region with the ground truth)
    jaccard = np.sum(np.logical_and(predicted_mask, gt_mask)) / np.sum(np.logical_or(predicted_mask, gt_mask))
    return metric, jaccard


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
                            (1 / np.sqrt(sum_dist_square + 0.01)))


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
    fore_mask = img[mask == 3]
    bg_t_link, fg_t_link = t_link_calc(fore_mask, bgGMM, fgGMM)
    return -np.log(bg_t_link), -np.log(fg_t_link)


def t_link_calc(fg_mask, bgGMM, fgGMM):
    # Calculate the numerator and denominator for the t-link equations
    bg_x = fg_mask[:, np.newaxis] - bgGMM['means'][np.newaxis]
    fg_x = fg_mask[:, np.newaxis] - fgGMM['means'][np.newaxis]

    bg_left_fac = bgGMM['weights'] * np.sqrt(bgGMM['dets'])
    fg_left_fac = fgGMM['weights'] * np.sqrt(fgGMM['dets'])

    bg_E = bgGMM['inv_covs']
    fg_E = fgGMM['inv_covs']

    bg_xEx = np.zeros((fg_mask.shape[0]))
    fg_xEx = np.zeros((fg_mask.shape[0]))

    for i in range(bgGMM['n_components']):
        bg_xEx += compute_ut_a_u2(bg_x[:, i, :], bg_E[i], bg_left_fac[i])
    for i in range(fgGMM['n_components']):
        fg_xEx += compute_ut_a_u2(fg_x[:, i, :], fg_E[i], fg_left_fac[i])

    return bg_xEx, fg_xEx


def compute_ut_a_u2(x, E, left_factor):
    x_E_x = np.sum(x * np.dot(x, E), axis=1)
    calc = left_factor * np.exp(-0.5 * x_E_x)
    return calc


# TODO: maybe vectorize the function
def init_graph(weights):
    global g
    g = ig.Graph()
    add_nodes()
    add_n_links_edges(weights)


def add_nodes():
    global g, row, col
    # add nods to graph
    for i in range(row):
        for j in range(col):
            vertex_id = vertex_name(i, j)
            g.add_vertex(vertex_id)
    # add source and sink
    g.add_vertex(FG)
    g.add_vertex(BG)


def gmm_fill_init(GMM, kmeans, pixels):
    GMM['n_components'] = n_components
    GMM['means'] = kmeans.cluster_centers_
    GMM['covs'] = np.array([np.cov(pixels[kmeans.labels_ == i].T) for i in range(n_components)])
    GMM['dets'] = np.array([np.linalg.det(GMM['covs'][i]) for i in range(n_components)])
    GMM['dets'] = GMM['dets']
    GMM['covs'] += np.eye(3) * 1e-6
    GMM['inv_covs'] = np.linalg.inv(GMM['covs'])


def gmm_init():
    GMM = {
        'n_components': n_components,
        'weights': np.full(n_components, 1 / n_components),
        'means': np.zeros((n_components, 3)),
        'covs': np.zeros((n_components, 3, 3)),
        'dets': np.zeros(n_components),
        'inv_covs': np.zeros((n_components, 3, 3))}
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
    parser.add_argument('--input_name', type=str, default='book', help='name of image from the course files')
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
