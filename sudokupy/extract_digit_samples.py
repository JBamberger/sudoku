import os

import cv2 as cv
import numpy as np
import torch
from torch.nn import functional as F

from utils import read_ground_truth


def create_rect_grid2(image, coords, in_size, out_size):
    H_in, W_in = in_size
    H_out, W_out = out_size

    # gauss = gaussian_shaped_labels(5, [21, 21]).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(3, 3, 1, 1)
    # image = F.conv2d(image, gauss, padding=10)

    # [N, H_out, W_out, 2]
    grid = torch.zeros((1, H_out, W_out, 2))
    ul = coords[0, :]
    ur = coords[1, :]
    br = coords[2, :]
    bl = coords[3, :]

    x1 = (torch.linspace(ul[0], ur[0], steps=W_out) / W_in - 0.5) * 2
    x2 = (torch.linspace(bl[0], br[0], steps=W_out) / W_in - 0.5) * 2
    y1 = (torch.linspace(ul[1], bl[1], steps=H_out) / H_in - 0.5) * 2
    y2 = (torch.linspace(ur[1], br[1], steps=H_out) / H_in - 0.5) * 2

    for i, p in enumerate(np.linspace(0, 1, H_out)):
        grid[0, i, :, 0] = (1 - p) * x1 + p * x2
    for i, p in enumerate(np.linspace(0, 1, W_out)):
        grid[0, :, i, 1] = (1 - p) * y1 + p * y2
    return grid


def extract_digit_samples():
    gt_file = os.path.abspath('ground_truth_new.csv')
    out_dir = os.path.abspath('extracted_digits')
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for file_path, coords in read_ground_truth(gt_file):
        sudoku_ori = cv.imread(file_path, cv.IMREAD_COLOR)

        sudoku = torch.from_numpy(sudoku_ori).permute((2, 0, 1)).unsqueeze(0).float()

        coords = coords.astype(np.int32)

        grid = create_rect_grid2(sudoku, coords, in_size=sudoku.shape[2:], out_size=[512, 512])
        aligned_sudoku = F.grid_sample(sudoku, grid, mode='bilinear', align_corners=False)

        poly = cv.polylines(sudoku_ori.copy(), [coords], True, (0, 255, 0), thickness=5)
        poly = cv.resize(poly, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        cv.imshow('Poly', poly)

        img = aligned_sudoku.squeeze(0).permute((1, 2, 0)).numpy().astype(np.uint8)
        cv.imshow('window', img)

        cv.waitKey()


if __name__ == '__main__':
    extract_digit_samples()
