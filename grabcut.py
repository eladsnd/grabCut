import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import argparse
import igraph as ig
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

n_components = 5

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

WEIGHT = 'weight'

global g, beta, row, col, number_of_existing_edges, prev_energy, prev_diff, K, number_of_edges_to_delete


# k = infinity


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    global prev_energy, prev_diff
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
    print("Initialization Time:_____", INITEND - INIT)
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
        if i == 7:
            break
        # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask):
    global g, beta, row, col, prev_energy, prev_diff, K, number_of_edges_to_delete
    # init
    row, col = img.shape[:2]
    d_adjacent, d_below, diag1_n_link, diag2_n_link = calc_beta_and_n_link(img)
    weights = np.concatenate((d_adjacent, d_below, diag1_n_link, diag2_n_link))
    K = max(weights)
    init_graph(weights)
    number_of_edges_to_delete = 0
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
    gmm_fill_init(fgGMM, fg_kmeans, fg_pixels)

    gmm_fill_init(bgGMM, bg_kmeans, bg_pixels)

    return fgGMM, bgGMM, weights


def update_GMMs(img, mask, bgGMM, fgGMM):
    # Get the pixels of the foreground and the background from the mask
    fg_pixels = img[mask > 0].reshape(-1, 3)
    bg_pixels = img[mask == 0].reshape(-1, 3)

    fg_components = np.zeros((fg_pixels.shape[0], fgGMM['n_components']))
    bg_components = np.zeros((bg_pixels.shape[0], bgGMM['n_components']))

    diff_back = bg_pixels[:, np.newaxis] - bgGMM['means'][np.newaxis]
    diff_fore = fg_pixels[:, np.newaxis] - fgGMM['means'][np.newaxis]

    for i in range(fgGMM['n_components']):
        fg_components[:, i] = multivariate_normal.pdf(fg_pixels, fgGMM["means"][i], fgGMM["covs"][i],
                                                      allow_singular=True)
        bg_components[:, i] = multivariate_normal.pdf(bg_pixels, bgGMM["means"][i], bgGMM["covs"][i],
                                                      allow_singular=True)
        # fg_left_factor = fgGMM['dets'][i]
        # if fg_left_factor == 0:
        #     fg_left_factor = 1
        # fg_components[:, i] = f_x(diff_fore[:, i], fgGMM["inv_covs"][i], fg_left_factor)

    # for i in range(bgGMM['n_components']):
    #     bg_left_factor = bgGMM['dets'][i]
    #     if bg_left_factor == 0:
    #         bg_left_factor = 1
    #     bg_components[:, i] = f_x(diff_back[:, i], bgGMM["inv_covs"][i], bg_left_factor)

    fg_new_indices = np.argmax(fg_components, axis=1)
    bg_new_indices = np.argmax(bg_components, axis=1)
    gmm_fill_new(fgGMM, fg_pixels, fg_new_indices)
    gmm_fill_new(bgGMM, bg_pixels, bg_new_indices)

    return bgGMM, fgGMM


def f_x(x, E, left_factor):
    x_E_x = np.sum(x * np.dot(x, E), axis=1)
    calc = left_factor * np.exp(-0.5 * x_E_x)
    return calc


def gmm_fill_new(GMM, pixels, new_indices):
    num_components = 0
    for i in range(GMM['n_components']):
        gmm_pixels = pixels[np.where(new_indices == i)]
        if gmm_pixels.shape[0] != 0:
            GMM['means'][i] = np.mean(gmm_pixels, axis=0)
            GMM['covs'][i] = np.cov(gmm_pixels.T)
            GMM['dets'][i] = np.linalg.det(GMM['covs'][i])
            GMM['covs'][i] += np.eye(3) * 1e-6
            GMM['inv_covs'][i] = np.linalg.inv(GMM['covs'][i])
            # num_components += 1
    # GMM['n_components'] = num_components
    # GMM['weights'] = np.ones(num_components) / num_components
    # GMM['means'] = GMM['means'][:num_components]
    # GMM['covs'] = GMM['covs'][:num_components]
    # GMM['dets'] = GMM['dets'][:num_components]
    # GMM['inv_covs'] = GMM['inv_covs'][:num_components]


def add_t_links(t_s2, t_t2, mask):
    global g, number_of_existing_edges, number_of_edges_to_delete

    if number_of_edges_to_delete > 0:
        edges_to_delete = g.es[-number_of_edges_to_delete:]
        g.delete_edges(edges_to_delete)

    unknown, background = map_mask_to_img(mask)

    g.add_edges(zip(['s'] * len(unknown), unknown))
    g.add_edges(zip(['t'] * len(unknown), unknown))

    g.add_edges(zip(['s'] * len(background), background))
    g.add_edges(zip(['t'] * len(background), background))

    s_background = K * np.ones(len(background))
    t_background = np.zeros(len(background))

    t = np.concatenate((t_s2, t_t2, s_background, t_background))
    g.es[number_of_existing_edges:][WEIGHT] = t
    number_of_edges_to_delete = len(t)
    # return g


# TODO: finish this function
def calculate_mincut(img, mask, bgGMM, fgGMM):
    t_link_s, t_link_t = t_link(img, mask, bgGMM, fgGMM)

    add_t_links(t_link_s, t_link_t, mask)

    min_cut = g.st_mincut(source='s', target='t', capacity=WEIGHT)

    energy = min_cut.value
    min_cut = min_cut.partition
    # s = min_cut[0]
    # t = min_cut[1]
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
    foreground = np.array(mincut_sets[0][1:]) - 2

    # create a mask with the same size as the image
    mask = np.zeros((row, col)).flatten()

    mask[foreground] = 3

    mask = mask.reshape((row, col))

    return mask


def check_convergence(energy):
    global prev_energy, prev_diff
    curr_diff = abs(energy - prev_energy)
    print("curr_diff: ", curr_diff)
    if curr_diff < 0.1:
        convergence = True
        return convergence
    prev_diff = curr_diff
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
                            np.where((sum_dist_square > 0), (1 / np.sqrt(sum_dist_square)), 0))


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

    left_factor_back = bgGMM['weights'] * bgGMM['dets']
    left_factor_fore = fgGMM['weights'] * fgGMM['dets']
    bg_E = bgGMM['inv_covs']
    fg_E = fgGMM['inv_covs']

    ut_a_u_back2 = np.zeros((img_masked.shape[0]))
    ut_a_u_fore2 = np.zeros((img_masked.shape[0]))
    for i in range(bgGMM['n_components']):
        ut_a_u_back2 += compute_ut_a_u2(diff_back[:, i, :], bg_E[i], left_factor_back[i])
    for i in range(fgGMM['n_components']):
        ut_a_u_fore2 += compute_ut_a_u2(diff_fore[:, i, :], fg_E[i], left_factor_fore[i])

    return ut_a_u_back2, ut_a_u_fore2


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
    g.add_vertex('s')
    g.add_vertex('t')


def gmm_fill_init(GMM, kmeans, pixels):
    GMM['n_components'] = n_components
    GMM['means'] = kmeans.cluster_centers_
    GMM['covs'] = np.array([np.cov(pixels[kmeans.labels_ == i].T) for i in range(n_components)])
    GMM['dets'] = np.array([np.linalg.det(GMM['covs'][i]) for i in range(n_components)])
    GMM['dets'] = 1 / np.sqrt(GMM['dets'])
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
