import cv2 as cv
import numpy as np
import torch
from torch.nn import functional as F


def read_ground_truth(gt_file):
    with open(gt_file, 'r', encoding='utf8') as f:
        annotations = [tuple(map(str.strip, line.split(','))) for line in f]
        annotations = [(annot[0], list(map(int, annot[1:]))) for annot in annotations]
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