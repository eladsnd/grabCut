import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse
from grabcut import grabcut


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(1, 1)
    mat_D.setdiag(-4)
    mat_D.setdiag(1, -1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(1, -1 * m)
    mat_A.setdiag(1, 1 * m)

    return mat_A


def reshape_image(img_src, img_mask, center, img_tgt):
    # x_odd = img_tgt.shape[0] % 2
    # y_odd = img_tgt.shape[1] % 2
    x_odd = 0
    y_odd = 0
    padding_x = abs(int(center[1] - (img_src.shape[0] / 2)))
    padding_y = abs(int(center[0] - (img_src.shape[1] / 2)))

    if img_tgt.shape[0] > img_src.shape[0] + padding_x * 2:
        x_odd = 1
    if img_tgt.shape[1] > img_src.shape[1] + padding_y * 2:
        y_odd = 1

    if img_tgt.shape[0] < img_src.shape[0] + padding_x * 2:
        x_odd -= 1
    if img_tgt.shape[1] < img_src.shape[1] + padding_y * 2:
        y_odd -= 1

    pad_src = np.pad(img_src, ((padding_x, padding_x + x_odd), (padding_y, padding_y + y_odd), (0, 0)),
                     mode='constant', constant_values=0)

    pad_mask = np.pad(img_mask, ((padding_x, padding_x + x_odd), (padding_y, padding_y + y_odd)),
                      mode='constant', constant_values=0)

    return pad_src, pad_mask


def get_range(img_target):
    y_max, x_max = img_target.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min
    return x_range, y_range, y_min, y_max, x_min, x_max


def poisson_blend(im_src, im_tgt, im_mask, center):
    # x_odd = 0 if img_tgt.shape[0] % 2 == 0 else 1
    # y_odd = 0 if img_tgt.shape[1] % 2` `== 0 else 1

    pad_src, pad_mask = reshape_image(im_src, im_mask, center, im_tgt)

    A = laplacian_matrix(pad_mask.shape[0], pad_mask.shape[1])
    lap = A.tocsc()

    x_range, y_range, y_min, y_max, x_min, x_max = get_range(im_tgt)

    mask = pad_mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1

    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                if k + x_range < A.shape[0]:
                    A[k, k] = 1
                    A[k, k + 1] = 0
                    A[k, k - 1] = 0
                    A[k, k + x_range] = 0
                    A[k, k - x_range] = 0
    A = A.tocsc()

    mask_flat = mask.flatten()
    for channel in range(pad_src.shape[2]):
        source_flat = pad_src[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = im_tgt[y_min:y_max, x_min:x_max, channel].flatten()

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        b = lap.dot(source_flat) * alpha

        # outside the mask:
        # f = t
        b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(A, b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')

        im_tgt[y_min:y_max, x_min:x_max, channel] = x

    im_blend = im_tgt
    return im_blend


def run_on_grabcut_masks():
    images = ['llama', 'memorial', 'sheep', 'stone2', 'fullmoon', 'teddy']
    images = ['fullmoon', 'teddy', 'stone2']
    images = ['sheep']

    bgs = ['grass_mountains.jpeg', 'table.jpg', 'wall.jpg']
    imgs_path = './data/imgs/'
    masks_path = './data/masks/'
    bgs_path = './data/bg/'
    for img in images:
        for bg in bgs:
            for i in range(1):
                try:
                    im_tgt = cv2.imread(bgs_path + bg, cv2.IMREAD_COLOR)
                    im_src = cv2.imread(imgs_path + img + '.jpg', cv2.IMREAD_COLOR)
                    im_mask = cv2.imread(masks_path + img + '.bmp', cv2.IMREAD_GRAYSCALE)
                    img_mask_seg_GT = cv2.imread('./data/seg_GT/' + img + '.bmp', cv2.IMREAD_GRAYSCALE)
                    if im_tgt.shape[0] < im_src.shape[0] or im_tgt.shape[1] < im_src.shape[1]:
                        continue
                    if i == 0:
                        im_blend = poisson_blend(im_src, im_tgt, im_mask, (im_tgt.shape[1] / 2, im_tgt.shape[0] / 2))
                        if not os.path.exists('data/results/'):
                            os.makedirs('data/results')
                        cv2.imwrite('data/results/' + img + '_' + bg.split('.')[0] + '_our_mask.jpg', im_blend)
                    if i == 1:
                        im_blend = poisson_blend(im_src, im_tgt, img_mask_seg_GT,
                                                 (im_tgt.shape[1] / 2, im_tgt.shape[0] / 2))
                        cv2.imwrite('data/results/' + img + '_' + bg.split('.')[0] + '_seg_GT.jpg', im_blend)
                    print('Saved ' + img + '_' + bg.split('.')[0] + '.jpg')
                except(ValueError, TypeError, AttributeError):
                    print('Failed ' + img + '_' + bg.split('.')[0] + '.jpg')


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana2.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/wall.jpg', help='mask file path')
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
