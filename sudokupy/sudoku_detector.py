from dataclasses import dataclass

import cv2 as cv
import numpy as np

from classifier.classifier import Net
from detection_utils import in_resize, detect_sudoku, unwarp_patch, pad_contour, SudokuNotFoundException, \
    extract_cells
from solver.sudoku_solver import solve_sudoku


@dataclass
class SudokuDetection:
    norm_scale_sudoku: np.ndarray
    norm_scale_factor: float

    pred_location: np.ndarray = None
    unwarp_sudoku: np.ndarray = None
    unwarp_grid: np.ndarray = None

    cell_images: np.ndarray = None
    cell_coords: np.ndarray = None

    occupied_cells: np.ndarray = None
    cell_values: np.ndarray = None

    solved_sudoku: np.ndarray = None


class SudokuDetector:
    def __init__(self):
        self.classifier = Net(size=64)
        self.classifier.load()

    def detect(self, frame: np.ndarray) -> SudokuDetection:
        sudoku_img_org = frame.copy()

        # Scaling such that the longer side is 1024 px long
        input_downscale, sudoku_img = in_resize(sudoku_img_org)

        det = SudokuDetection(sudoku_img, input_downscale)

        try:
            det.pred_location = detect_sudoku(det.norm_scale_sudoku)
        except SudokuNotFoundException:
            return det

        # add padding to compensate for bounding boxes that fit too tight
        padded_location = pad_contour(det.pred_location, padding=0)

        # unwarp from original image for best warp quality
        scaled_locations = padded_location / det.norm_scale_factor
        sudoku_coarse_unwarp, crop_grid = unwarp_patch(sudoku_img_org, scaled_locations, return_grid=True)
        crop_grid *= det.norm_scale_factor

        det.unwarp_sudoku = sudoku_coarse_unwarp
        det.unwarp_grid = crop_grid

        cell_images, cell_coords = extract_cells(sudoku_coarse_unwarp)
        cell_images = cell_images.astype(np.uint8)  # TODO: integrate into extract_cells
        det.cell_images = cell_images
        det.cell_coords = cell_coords

        out = np.zeros((81,), dtype=np.int32)
        for i in range(81):
            cell_patch = cell_images[i, :, :, :]

            gray_cell = cv.cvtColor(cell_patch, cv.COLOR_BGR2GRAY)

            pad = 6
            _, pt = cv.threshold(gray_cell[pad:-pad, pad:-pad], 100, 255, cv.THRESH_BINARY_INV)
            if np.count_nonzero(pt) > 100 and np.var(gray_cell) > 500:
                out[i] = self.classifier.classify(gray_cell)
            else:
                out[i] = 0

        det.occupied_cells = out != 0
        det.cell_values = out.reshape((9, 9))

        solved_sudoku = solve_sudoku(det.cell_values.tolist())
        solved_sudoku = None if solved_sudoku is None else np.array(solved_sudoku)
        det.solved_sudoku = solved_sudoku

        return det
