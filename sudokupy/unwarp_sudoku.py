import os
import time

import cv2 as cv
import numpy as np

from classifier import Net
from detection_utils import in_resize, detect_sudoku, unwarp_patch, pad_contour, SudokuNotFoundException, \
    extract_cells
from solver.sudoku_solver import solve_sudoku
from utils import rotation_correction, read_ground_truth, show


def save_cell_patch(cell_patch, classification):
    gray_patch = cv.cvtColor(cell_patch, cv.COLOR_BGR2GRAY)
    _, pt = cv.threshold(gray_patch, 100, 255, cv.THRESH_BINARY_INV)
    nnz = np.count_nonzero(pt)

    sudoku_basename = os.path.splitext(os.path.basename(file_path))[0]
    cell_path = os.path.join('extracted_digits', f'{classification}_{nnz:08d}_{sudoku_basename}_{i}.jpg')
    cv.imwrite(cell_path, cell_patch)


gt_annoatations = read_ground_truth(np.os.path.abspath('ground_truth_new.csv'))
digit_classifier = Net(size=64)
digit_classifier.load()

for sudoku_index, (file_path, gt_coords) in enumerate(gt_annoatations):
    if sudoku_index < 0:
        continue

    start = time.time()
    sudoku_img_org = cv.imread(file_path)

    # Ensure that the sudoku is always rotated by at most 45 deg in either direction.
    sudoku_img_org, gt_coords = rotation_correction(sudoku_img_org, gt_coords)

    # Scaling such that the longer side is 1024 px long
    input_downscale, sudoku_img = in_resize(sudoku_img_org)
    image_center = (int(round(sudoku_img.shape[1] / 2)), int(round(sudoku_img.shape[0] / 2)))

    # Visualization canvas
    canvas = sudoku_img.copy()

    # Draw gt rect
    gt_coords = np.array(gt_coords).reshape(4, 2)
    gt_coords = (gt_coords * input_downscale).astype(np.int32)
    cv.polylines(canvas, [gt_coords], True, (0, 255, 0), thickness=3)

    # draw image center
    cv.drawMarker(canvas, image_center, (0, 0, 255))

    try:
        pred_location = detect_sudoku(sudoku_img)
    except SudokuNotFoundException as e:
        print(f'Failed to detect sudoku. {e}')

        cv.putText(canvas, 'Detection failed!', image_center, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
        show(canvas, name='Bounds')
        continue

    # add padding to compensate for bounding boxes that fit too tight
    padded_location = pad_contour(pred_location, padding=0)

    # unwarp from original image for best warp quality
    scaled_locations = padded_location / input_downscale
    sudoku_coarse_unwarp, crop_grid = unwarp_patch(sudoku_img_org, scaled_locations, return_grid=True)
    crop_grid *= input_downscale

    cell_images, cell_coords = extract_cells(sudoku_coarse_unwarp)
    cell_images = cell_images.astype(np.uint8)

    out = np.zeros((81,), dtype=np.int32)
    for i in range(81):
        cell_patch = cell_images[i, :, :, :]

        gray_cell = cv.cvtColor(cell_patch, cv.COLOR_BGR2GRAY)
        # deskewed_cell = deskew(gray_cell)

        pad = 6
        _, pt = cv.threshold(gray_cell[pad:-pad, pad:-pad], 100, 255, cv.THRESH_BINARY_INV)
        # print(f'{np.count_nonzero(pt): 4d} {np.var(gray_cell): 5.02f}')
        if np.count_nonzero(pt) > 100 and np.var(gray_cell) > 500:
            # out[i] = classify_digit(gray_cell)
            out[i] = digit_classifier.classify(gray_cell)
        else:
            out[i] = 0

        # save_cell_patch(cell_patch, out[i])

    is_empty_cell = out == 0
    out = out.reshape((9, 9))
    print(out)

    solved_sudoku = solve_sudoku(out.tolist())
    solved_sudoku = out if solved_sudoku is None else np.array(solved_sudoku)
    # print(solved_sudoku)

    solved_sudoku = solved_sudoku.flatten()
    for i in range(81):
        cell_center = cell_coords[i, :, :].mean(0).astype(np.int32)
        cell_center = crop_grid[0, cell_center[1].item(), cell_center[0].item(), :].astype(np.int32)
        cell_center = (cell_center[0].item(), cell_center[1].item())

        color = (0, 0, 255) if is_empty_cell[i] else (0, 255, 0)
        cv.putText(canvas, str(solved_sudoku[i]), cell_center, cv.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        # cv.drawMarker(canvas, cell_center, (0, 255, 0))

    cv.polylines(canvas, [pred_location], True, (255, 0, 0), thickness=3)
    cv.line(canvas, tuple(pred_location[0, :]), tuple(pred_location[1, :]), (255, 0, 255), thickness=10)
    cv.line(canvas, tuple(pred_location[1, :]), tuple(pred_location[2, :]), (255, 255, 0), thickness=10)

    for i in range(4):
        cv.putText(canvas, 'ABCD'[i], tuple(pred_location[i, :]),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    print('Took ', time.time() - start)
    show(canvas, name='Bounds')

# sudoku_brightness_normalized

# kernel_size = 25
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
#
# closing = cv.morphologyEx(sudoku_gray, cv.MORPH_CLOSE, kernel)
# show(closing)
#
# # opening = cv.morphologyEx(sudoku_gray, cv.MORPH_OPEN, kernel)
# # show(opening)
#
# sudoku_corrected = (sudoku_gray / closing * 255).astype(np.uint8)
# show(sudoku_corrected)
#
# blockSize = 2
# apertureSize = 3
# k = 0.04
#
# dst = cv.cornerHarris(sudoku_corrected, blockSize, apertureSize, k)
#
# dst_norm = np.empty(dst.shape, dtype=np.float32)
# cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
# dst_norm_scaled = cv.convertScaleAbs(dst_norm)
#
# # for i in range(dst_norm.shape[0]):
# #     for j in range(dst_norm.shape[1]):
# #         if int(dst_norm[i, j]) > 100:
# #             cv.circle(dst_norm_scaled, (j, i), 5, (0), 2)
#
# show(dst_norm_scaled)
#
# blurred = cv.GaussianBlur(sudoku_corrected, (5, 5), 0)
# show(blurred)
#
# thresh, sudoku_bin = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# show(sudoku_bin)
