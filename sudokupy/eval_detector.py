import os

import cv2 as cv
import numpy as np

from sudoku_detector import SudokuDetector
from utils import read_ground_truth


def eval_detector():
    annotations = read_ground_truth(os.path.abspath('ground_truth_new.csv'))
    detector = SudokuDetector()

    for file_path, coords in annotations:
        image = cv.imread(file_path, cv.IMREAD_COLOR)
        sudoku_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sudoku_ori = image.copy()
        sudoku = detector.detect_sudoku(sudoku_gray)

        poly = cv.polylines(sudoku_ori.copy(), [coords], True, (0, 255, 0), thickness=5)

        pred_coords = sudoku.corners.as_array().astype(np.int32)
        poly = cv.polylines(poly, [pred_coords], True, (0, 0, 255), thickness=5)

        print(coords)
        print(pred_coords)

        poly = cv.resize(poly, None, fx=0.4, fy=0.4, interpolation=cv.INTER_AREA)
        cv.imshow('Poly', poly)

        cv.waitKey()


if __name__ == '__main__':
    eval_detector()
