import math
from typing import Tuple

import cv2 as cv
import numpy as np
import torch
from torch.nn import functional as F


def read_ground_truth(gt_file):
    with open(gt_file, 'r', encoding='utf8') as f:
        annotations = []
        for line in f:
            fields = tuple(map(str.strip, line.split(',')))
            coords = np.array(list(map(int, fields[1:]))).reshape(4, 2).astype(np.int32)
            pair = (fields[0], coords)
            annotations.append(pair)
    return annotations


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


CCW90 = np.array([[0, -1], [1, 0]])
CCW180 = np.array([[-1, 0], [0, -1]])
CCW270 = np.array([[0, 1], [-1, 0]])


def oriented_angle(x1, x2, y1, y2):
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    angle = math.atan2(det, dot)
    return angle


def rotation_correction(img, coords) -> Tuple[np.ndarray, np.ndarray]:
    # angle between upper border (reading direction) and e1
    x1, y1 = coords[1, :] - coords[0, :]
    x2, y2 = 1, 0
    angle = oriented_angle(x1, x2, y1, y2)
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
