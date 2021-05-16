import time

import cv2 as cv
import numpy as np

import config
from sudoku_detector import SudokuDetector
from utils import rotation_correction, read_ground_truth, show

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)

gt_annoatations = read_ground_truth(config.sudokus_gt_path)

detector = SudokuDetector()

for sudoku_index, (file_path, gt_coords) in enumerate(gt_annoatations):
    if sudoku_index < 0:
        continue

    start = time.time()
    sudoku_img_org = cv.imread(file_path)

    # Ensure that the sudoku is always rotated by at most 45 deg in either direction.
    sudoku_img_org, gt_coords = rotation_correction(sudoku_img_org, gt_coords)

    det = detector.detect(sudoku_img_org)

    # Visualization canvas
    canvas = det.norm_scale_sudoku.copy()
    image_center = (int(round(canvas.shape[1] / 2)), int(round(canvas.shape[0] / 2)))

    # Draw gt rect
    gt_coords = np.array(gt_coords).reshape(4, 2)
    gt_coords = (gt_coords * det.norm_scale_factor).astype(np.int32)
    cv.polylines(canvas, [gt_coords], True, GREEN, thickness=3)

    # draw image center
    cv.drawMarker(canvas, image_center, RED)

    if det.pred_location is None:
        print(f'Failed to detect sudoku.')

        cv.putText(canvas, 'Detection failed!', image_center, cv.FONT_HERSHEY_SIMPLEX, 2, RED, thickness=3)
    else:
        bounds = det.pred_location

        solved_sudoku = det.cell_values if det.solved_sudoku is None else det.solved_sudoku

        # Cell values
        solved_sudoku = solved_sudoku.flatten()
        for i in range(81):
            cell_center = det.cell_coords[i, :, :].mean(0).astype(np.int32)
            cell_center = det.unwarp_grid[0, cell_center[1].item(), cell_center[0].item(), :].astype(np.int32)
            cell_center = (cell_center[0].item(), cell_center[1].item())

            color = GREEN if det.occupied_cells[i] else RED
            cv.putText(canvas, str(solved_sudoku[i]), cell_center, cv.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

        # Sudoku detection bounds
        cv.polylines(canvas, [bounds], True, BLUE, thickness=3)
        cv.line(canvas, tuple(bounds[0, :]), tuple(bounds[1, :]), MAGENTA, thickness=10)
        cv.line(canvas, tuple(bounds[1, :]), tuple(bounds[2, :]), CYAN, thickness=10)

        # Sudoku corner labels
        for i in range(4):
            cv.putText(canvas, 'ABCD'[i], tuple(bounds[i, :]), cv.FONT_HERSHEY_SIMPLEX, 1, RED, thickness=2)

    print('Took ', time.time() - start)
    show(canvas, name='Sudoku')

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
