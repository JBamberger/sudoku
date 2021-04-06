import math

import numpy as np
import cv2 as cv
import torch
from torch.nn import functional as F


# from unwarp_sudoku import show


def squared_p2p_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def p2p_dist(p1, p2):
    return math.sqrt(squared_p2p_dist(p1, p2))


def in_resize(image, long_side=1024):
    h, w = image.shape[:2]

    # scale the longer side to long_side
    scale = long_side / (h if h > w else w)

    return scale, cv.resize(image, None, fx=scale, fy=scale)


def detect_sudoku(sudoku_img):
    # conversion to gray
    sudoku_gray = cv.cvtColor(sudoku_img, cv.COLOR_BGR2GRAY)

    # Lightness normalization with morphological closing operation (basically subtracts background color)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(25, 25))
    closing = cv.morphologyEx(sudoku_gray, cv.MORPH_CLOSE, kernel)
    sudoku_gray = (sudoku_gray / closing * 255).astype(np.uint8)

    # sudoku_bin = cv.GaussianBlur(sudoku_gray, (5, 5), 0)
    # sudoku_bin = cv.adaptiveThreshold(
    #     sudoku_bin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize=21, C=127)

    # Inverse binarization with OTSU to find best threshold automatically (lines should be 1, background 0)
    threshold, sudoku_bin = cv.threshold(sudoku_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # sudoku_bin = cv.medianBlur(sudoku_bin, 5)

    # Dilation to enlarge the binarized structures slightly (fix holes in lines etc.)
    # Must be careful not to over-dilate, otherwise sudoku can merge with surroundings -> bad bbox/contour
    dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilated = cv.dilate(sudoku_bin, dilation_kernel)

    # show(dilated)

    image_center = (int(round(sudoku_img.shape[1] / 2)), int(round(sudoku_img.shape[0] / 2)))

    # Finding the largest contour (should be the sudoku)
    contours, hierarchy = cv.findContours(dilated, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_index = -1
    for i in range(len(contours)):
        contour = contours[i]

        # test if the image center is within the proposed region
        polytest = cv.pointPolygonTest(contour, image_center, measureDist=False)
        if polytest < 0:
            continue

        points = cv.boxPoints(cv.minAreaRect(contour))
        a, b, c, d = points
        d1 = squared_p2p_dist(a, b)
        d2 = squared_p2p_dist(a, d)

        square_thresh = (d1 + d2) / 2 * 0.5
        square_diff = abs(d1 - d2)
        if square_diff > square_thresh:
            # print(f'Not square {square_diff} {square_thresh}')
            continue

        # test if the contour is large enough
        contour_area = cv.contourArea(contour, oriented=False)
        if contour_area > max_area:
            max_area = contour_area
            max_index = i

    if max_index < 0:
        raise RuntimeError('Sudoku not found.')

    points = cv.boxPoints(cv.minAreaRect(contours[max_index]))
    #     a, b, c, d = points
    #     d1 = squared_p2p_dist(a, b)
    #     d2 = squared_p2p_dist(a, d)
    #     print(d1, ' ', d2, ' ', abs(d1 - d2))

    rect = np.array(points).reshape(4, 2).astype(np.int32)

    # x, y, w, h = cv.boundingRect(contours[max_index])

    # for i in range(len(contours)):
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     cv.drawContours(sudoku_img, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # show(sudoku_img, name='contours')

    # img = cv.rectangle(sudoku_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)
    # show(img, name='Bounds')

    return rect


def unwarp_patch(image, poly_coords, out_size=(1024, 1024), return_grid=False):
    h, w, _ = image.shape
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    grid = torch.tensor(poly_coords).float().view(4, 2)

    # Swap the last two coordinates to obtain a valid grid when reshaping
    grid = grid[[0, 1, 3, 2]]

    # Transform coordinates to range [-1, 1]
    grid[:, 0] /= w
    grid[:, 1] /= h
    grid -= 0.5
    grid *= 2.0

    # Interpolate grid to full output size
    grid = grid.view(1, 2, 2, 2).permute(0, 3, 1, 2)  # order as [1, 2, H, W]
    grid = F.interpolate(grid, out_size, mode='bilinear', align_corners=True)

    # compute interpolated output image
    grid = grid.permute(0, 2, 3, 1)  # Order as [1, H, W, 2]
    aligned_img = F.grid_sample(img_tensor, grid, mode='bilinear', align_corners=False)

    # back to numpy uint8
    interp_img = aligned_img.squeeze(0).permute(1, 2, 0).to(dtype=torch.uint8).numpy()

    if return_grid:
        grid /= 2.0
        grid += 0.5
        grid[:, :, :, 0] *= w
        grid[:, :, :, 1] *= h

        return interp_img, grid.numpy()

    return interp_img


def pad_contour(image, coords, padding=15):
    img = np.zeros(image.shape[:2], dtype=np.uint8)

    cv.fillPoly(img, [coords], color=(255,))

    dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (padding * 2, padding * 2))
    dilated = cv.dilate(img, dilation_kernel)

    contours, hierarchy = cv.findContours(dilated, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_index = 0
    for i in range(len(contours)):
        contour = contours[i]
        contour_area = cv.contourArea(contour, oriented=False)
        if contour_area > max_area:
            max_area = contour_area
            max_index = i

    points = cv.boxPoints(cv.minAreaRect(contours[max_index]))
    rect = np.array(points).reshape(4, 2).astype(np.int32)
    return rect
