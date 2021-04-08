import math

import numpy as np
import cv2 as cv
import torch
from torch.nn import functional as F

from utils import normalize_rect_orientation, squared_p2p_dist, p2p_dist, thresh_savoula, show


class SudokuNotFoundException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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

    # Finding the largest contour (should be the sudoku)
    contours, _ = cv.findContours(dilated, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    points = locate_sudoku_contour(contours, sudoku_img)

    return points


def locate_sudoku_contour(contours, sudoku_img):
    max_area = 0
    max_index = -1
    image_center = (int(round(sudoku_img.shape[1] / 2)), int(round(sudoku_img.shape[0] / 2)))
    for i in range(len(contours)):
        contour = contours[i]

        # test if the image center is within the proposed region
        polytest = cv.pointPolygonTest(contour, image_center, measureDist=False)
        if polytest < 0:
            continue

        # test if the contour is approximately square
        points = cv.boxPoints(cv.minAreaRect(contour))
        a, b, c, d = points
        d1 = squared_p2p_dist(a, b)
        d2 = squared_p2p_dist(a, d)
        square_thresh = (d1 + d2) / 2 * 0.5
        square_diff = abs(d1 - d2)
        if square_diff > square_thresh:
            continue

        # test if the contour is large enough
        contour_area = cv.contourArea(contour, oriented=False)
        if contour_area > max_area:
            max_area = contour_area
            max_index = i

    if max_index < 0:
        raise SudokuNotFoundException('Sudoku not found.')

    max_ct = contours[max_index]

    points = approx_quad(max_ct)

    return points


def approx_quad(points, normalize_orientation=True):
    """
    Approximates a contour or points with four corner points.

    :param points: points of the shape [N,1,2]
    :param normalize_orientation: True to normalize the orientation, i.e. first point is upper left, CW order.
    :return: quad corner points of shape [4, 2]
    """
    epsilon = 0.1 * cv.arcLength(points, closed=True)
    points = cv.approxPolyDP(points, epsilon, closed=True)

    if points.shape[0] != 4:
        print('Could not approx. sudoku shape with quad.')
        points = cv.minAreaRect(points)
        points = cv.boxPoints(points)
        points = np.array(points)

    points = points.reshape(4, 2).astype(np.int32)

    if normalize_orientation:
        points = normalize_rect_orientation(points)

    return points


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


def pad_contour(coords, padding=15):
    """
    Pad the coordinates, such that the point is projected on the line through the center and point.
    Negative values pad inwards, shrinking the contour and positive values pad outwards, growing the contour.

    :param coords: input coordinates of shape [N,2]
    :param padding: amount of padding to apply to the coordinates.
    :return: padded contour of shape [N,2]
    """
    center = coords.mean(axis=0)
    for i in range(len(coords)):
        vec = coords[i, :] - center
        sum_sq = np.sum(vec * vec)
        num = padding + np.sqrt(sum_sq)
        c = np.sqrt(num * num / sum_sq)

        coords[i, :] = center + c * vec

    return coords.astype(np.int32)


def extract_cells(image):
    sudoku = binarize_sudoku(image)

    cells = []
    contours, hierarchy = cv.findContours(sudoku, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contour_canvas = image.copy()
    for i in range(len(contours)):
        box = approx_quad(contours[i])

        area = cv.contourArea(box)
        if 80 * 80 <= area <= 120 * 120:
            center = box.mean(axis=0)
            cells.append((center, box))
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        contours[i] = box.reshape((-1, 1, 2))

        cv.drawContours(contour_canvas, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

    cell_to_node = compute_cell2node_mapping_fast(cells, contour_canvas)

    pad = 3
    patch_size = 64
    padded_size = (patch_size + 2 * pad, patch_size + 2 * pad)

    cell_coords = np.zeros((81, 4, 2))
    cell_patches = np.zeros((81, patch_size, patch_size, 3))
    for i in range(81):
        node_idx = cell_to_node.flatten()[i]
        if node_idx < 0:
            cell_coords[i, :, :] = -1
            continue

        coordinates = cells[node_idx][1]

        padded_cell_patch = unwarp_patch(image, coordinates, out_size=padded_size)

        cell_coords[i, :, :] = coordinates
        cell_patches[i, :, :, :] = padded_cell_patch[pad:pad + patch_size, pad:pad + patch_size]

    return cell_patches, cell_coords


def compute_cell2node_mapping_fast(cells, contour_canvas):
    step = 1024 / 9
    mapping = -np.ones((9, 9), dtype=np.int32)
    for cell_index, cell in enumerate(cells):
        cx, cy = cell[0]

        i = int(cx / step)
        j = int(cy / step)

        if not (0 <= i < 1024) or not (0 <= j < 1024):
            print(f'Position cannot be mapped. Out of range: ({i},{j})')

        if mapping[j][i] < 0:
            mapping[j][i] = cell_index
        else:
            print(f'Cell at ({i},{j}) already occupied by {mapping[j][i]}')

    for i in range(81):
        cell_idx = mapping.flatten()[i]
        if cell_idx < 0:
            continue
        cx, cy = cells[cell_idx][0]
        center = (int(cx), int(cy))
        cv.putText(contour_canvas, str(i), center, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)
        cv.drawMarker(contour_canvas, center, (0, 255, 255))

    show(contour_canvas, 'Cellmapping', True)

    print(mapping)

    return mapping


def compute_cell2node_mapping(cells, contour_canvas):
    # compute the cell graph
    edges = []
    for i in range(len(cells)):
        for j in range(len(cells)):
            if i == j:
                continue

            other = cells[j]

            p1 = cells[i][0]
            p2 = other[0]
            dist = p2p_dist(p1, p2)

            e1 = np.array([1, 0])
            diff_norm = (p1 - p2) / dist
            angle = np.arccos(np.dot(e1, diff_norm))

            dd = math.pi / 8

            mod_pi = angle % math.pi
            if mod_pi <= dd or math.pi - dd <= mod_pi:
                direction = 'h'
            elif math.pi / 2 - dd <= mod_pi <= math.pi / 2 + dd:
                direction = 'v'
            else:
                direction = None

            if direction is not None and 80 <= dist <= 115:
                edges.append((i, j, dist, angle, direction))

    # visualization
    for node_num, node in enumerate(cells):
        center = tuple(node[0].astype(np.int32))
        cv.drawMarker(contour_canvas, center, (0, 255, 0))
        cv.putText(contour_canvas, str(node_num), center, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    for edge in edges:
        direction = edge[4]
        p1 = tuple(cells[edge[0]][0].astype(np.int32))
        p2 = tuple(cells[edge[1]][0].astype(np.int32))

        color = (127, 0, 127) if direction == 'h' else (127, 127, 0)
        cv.line(contour_canvas, p1, p2, color)

    # find the first node
    first_node = 0
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge[0] != first_node:
                continue

            node = cells[first_node]
            other = cells[edge[1]]
            orientation = edge[4]
            if orientation == 'h' and node[0][0] > other[0][0] \
                    or orientation == 'v' and node[0][1] > other[0][1]:
                first_node = edge[1]
                changed = True
                continue

    cv.drawMarker(contour_canvas, tuple(cells[first_node][0].astype(np.int32)), (0, 255, 255),
                  markerSize=20, thickness=5)

    edgemap = {i: [edge for edge in filter(lambda e: e[0] == i, edges)] for i in range(len(cells))}
    cell_to_node = np.zeros((9, 9), dtype=np.int32)
    row_node = col_node = first_node
    for i in range(9):
        for j in range(9):
            cell_to_node[i, j] = col_node

            for edge in edgemap[col_node]:
                if edge[4] == 'h' and cells[col_node][0][0] < cells[edge[1]][0][0]:
                    col_node = edge[1]
            # TODO: Detect failures here, i.e. when col_nodes didn't change

        for edge in edgemap[row_node]:
            if edge[4] == 'v' and cells[row_node][0][1] < cells[edge[1]][0][1]:
                row_node = edge[1]
                col_node = row_node
        # TODO: Detect failures here, i.e. when row_node didn't change

    show(contour_canvas, name='Contours', no_wait=True)

    return cell_to_node


def binarize_sudoku(image):
    sudoku = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # show(sudoku, name='Gray', no_wait=True)

    # erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # sudoku = cv.erode(sudoku, erosion_kernel)
    # show(sudoku, name='Eroded', no_wait=True)

    # hist, bins = np.histogram(sudoku, bins=np.arange(0, 255))
    # plt.plot(hist)
    # plt.axvline(x=255-threshold)
    # plt.show()

    sudoku = cv.GaussianBlur(sudoku, (5, 5), 0)
    # show(sudoku, name='Blurred', no_wait=True)

    thresh = 'sav'
    if thresh == 'adaptive':
        sudoku = cv.adaptiveThreshold(sudoku, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 51, C=2)
    elif thresh == 'sav':
        sudoku = thresh_savoula(sudoku, window_size=51)
    else:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(21, 21))
        closing = cv.morphologyEx(sudoku, cv.MORPH_CLOSE, kernel)
        # show(closing, name='Closing', no_wait=True)

        sudoku = (sudoku / closing * 255).astype(np.uint8)
        # show(sudoku, name='GrayNorm', no_wait=True)

        threshold, _ = cv.threshold(sudoku, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # blur = cv.GaussianBlur(sudoku, (5, 5), 0)
        # _, sudoku2 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # show(sudoku2, name='BinarizedGaussian', no_wait=True)

        # threshold, sudoku = cv.threshold(sudoku, 127, 255, cv.THRESH_BINARY_INV)
        # sudoku = cv.adaptiveThreshold(sudoku, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 701, C=2)
        # sudoku = adaptiveThresh(sudoku, 50, 180)
        # threshold, _ = cv.threshold(sudoku, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # threshold, sudoku = cv.threshold(sudoku, threshold+20, 255, cv.THRESH_BINARY_INV)
    # print(f'Otsu`s thresh {threshold}')
    # show(sudoku, name='Binarized', no_wait=True)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(5, 5))
    sudoku = cv.dilate(sudoku, kernel)
    # show(sudoku, name='Binarized+dilated', no_wait=True)

    sudoku = cv.medianBlur(sudoku, 7)
    # show(sudoku, name='Binarized+Median', no_wait=True)

    # dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    # sudoku = cv.dilate(sudoku, dilation_kernel)
    # show(sudoku, name='Dilated', no_wait=True)

    # cv.waitKey()

    return sudoku
