import math
import os
from typing import Tuple

import cv2 as cv
import numpy as np
import torch
from skimage.filters import threshold_sauvola
from torch.nn import functional as F

import config


def read_ground_truth(gt_file):
    with open(gt_file, 'r', encoding='utf8') as f:
        annotations = []
        for line in f:
            fields = tuple(map(str.strip, line.split(',')))
            coords = np.array(list(map(int, fields[1:]))).reshape(4, 2).astype(np.int32)
            pair = (fields[0], coords)
            annotations.append(pair)
    return annotations


def squared_p2p_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def p2p_dist(p1, p2):
    return math.sqrt(squared_p2p_dist(p1, p2))


def gaussian2d(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2),
                       np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    return torch.from_numpy(g)


def compute_gradients(image):
    t = torch.tensor([[0, 0, 0], [1, 0, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    horiz = t.repeat(1, 3, 1, 1)
    vert = t.transpose(2, 3).repeat(1, 3, 1, 1)
    dx = F.conv2d(image, horiz, padding=1)
    dy = F.conv2d(image, vert, padding=1)
    mag = dx * dx + dy * dy
    mag = mag - mag.min()
    mag = mag / mag.max()
    mag = mag * 255
    mag = mag.to(dtype=torch.uint8)
    cv.imshow('Gradient magnitude', mag.squeeze(0).permute((1, 2, 0)).numpy())

    return dx, dy


# Matrices to rotate 2D coordinates
CCW90 = np.array([[0, -1], [1, 0]])
CCW180 = np.array([[-1, 0], [0, -1]])
CCW270 = np.array([[0, 1], [-1, 0]])


def vec_angle(x1, y1, x2, y2):
    v1_len = math.sqrt(x1 * x1 + y1 * y1)
    v2_len = math.sqrt(x2 * x2 + y2 * y2)
    n = v1_len * v2_len
    return math.acos(x1 * x2 / n + y1 * y2 / n)


def oriented_angle(x1, y1, x2, y2):
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    angle = math.atan2(det, dot)
    return angle


def rotation_correction(img, coords) -> Tuple[np.ndarray, np.ndarray]:
    # angle between upper border (reading direction) and e1
    x1, y1 = coords[1, :] - coords[0, :]
    x2, y2 = 1, 0
    angle = oriented_angle(x1, y1, x2, y2)
    angle_deg = angle / np.pi * 180

    if -45.0 <= angle_deg <= 45.0:
        return img, coords

    h, w = img.shape[:2]
    mid = np.array([w, h]).reshape(1, 2) / 2

    coords = coords - mid
    if -135.0 <= angle_deg < -45.0:
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        coords = np.matmul(coords, CCW90)
        mid = mid[:, ::-1]
    elif 45.0 < angle_deg <= 135.0:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        coords = np.matmul(coords, CCW270)
        mid = mid[:, ::-1]
    elif 135.0 < angle_deg or angle < -135.0:
        img = cv.rotate(img, cv.ROTATE_180)
        coords = np.matmul(coords, CCW180)

    coords = coords + mid
    coords = coords.astype(np.int32)

    # print(f'Angle: {angle_deg:>6.02f}')

    return img, coords


def normalize_rect_orientation(rect):
    assert rect.shape[0] == 4 and rect.shape[1] == 2

    # Ordering the indices by x and y coordinates, then taking the lower 2 values each
    lower_xy = np.concatenate([np.argsort(rect[:, 0])[:2], np.argsort(rect[:, 1])[:2]], axis=0)

    # Find the value which appears two times, i.e. being in the lower 2 values for x and y direction. There can only be
    # one, unless the rect is malformed (e.g. all points are the same) or at an angle of exactly 45deg. In the latter
    # case it is impossible to determine the correct orientation, thus an arbitrary choice is made.
    ul_idx = np.bincount(lower_xy).argmax()

    # Shift the coordinates to the correct position, such that the upper left corner is the first coordinate
    rect = np.roll(rect, shift=-ul_idx, axis=0)

    # oriented angle to e1
    angle = oriented_angle(rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1], 1, 0) / np.pi * 180
    # if the angle between the first rect side and the x axis is not between -45 and 45 the rectangle points are the
    # wrong way around. Flipping the coordinate order and shifting by 1 to bring the first coordinate back to pos 0
    # fixes the problem.
    if not (-45.0 <= angle <= 45.0):
        rect = np.roll(rect[::-1, :], shift=1, axis=0)

    return rect


def adaptiveThresh(image, lower, upper):
    assert lower < upper

    adapt = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 701, C=2)
    show(adapt, name='Binarized', no_wait=True)

    lowerimg = np.zeros_like(adapt)
    lowerimg[image < lower] = 255
    show(lowerimg, name='lower', no_wait=True)

    upperimg = np.zeros_like(adapt)
    upperimg[image > upper] = 255
    show(upperimg, name='upperimg')

    adapt[image < lower] = 255
    adapt[image > upper] = 0

    return adapt


def thresh_savoula(image, window_size=15, k=0.2):
    threshold = threshold_sauvola(image, window_size=window_size, k=k)
    image = (image < threshold).astype(np.uint8) * 255
    return image


def show(img, name='Image', no_wait=False):
    cv.imshow(name, cv.resize(img, (1024, int(img.shape[0] / (img.shape[1] / 1024))), interpolation=cv.INTER_AREA))
    if not no_wait:
        cv.waitKey()


def save_cell_patch(file_path, cell_num, cell_patch, classification):
    gray_patch = cv.cvtColor(cell_patch, cv.COLOR_BGR2GRAY)
    _, pt = cv.threshold(gray_patch, 100, 255, cv.THRESH_BINARY_INV)
    nnz = np.count_nonzero(pt)

    sudoku_basename = os.path.splitext(os.path.basename(file_path))[0]
    cell_path = os.path.join(config.extracted_digits_path,
                             f'{classification}_{nnz:08d}_{sudoku_basename}_{cell_num}.jpg')
    cv.imwrite(cell_path, cell_patch)
