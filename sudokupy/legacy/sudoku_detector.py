import math
from typing import Tuple, List, Callable

import cv2 as cv
import numpy as np

from sudoku_util import Sudoku, Quad

Line = Tuple[float, float]
Point = Tuple[float, float]
Size = Tuple[int, int]


class SudokuDetector:
    warp_size = (9 * 32, 9 * 32)
    cell_size = (28, 28)
    div_size = 2
    pcell_size = 28 + 2 * div_size

    def detect_sudoku(self, input_image: np.ndarray) -> Sudoku:
        sudoku = Sudoku(input_image)
        inverted = self.preprocess_image(sudoku)
        self.detect_sudoku_corners(sudoku, inverted)
        self.compute_unwarp_transform(sudoku)
        self.get_cells(sudoku)
        self.get_cell_contents(sudoku)
        return sudoku

    def preprocess_image(self, sudoku: Sudoku) -> np.ndarray:
        blurred = cv.GaussianBlur(sudoku.input, (11, 11), 3)
        inverted = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        inverted = cv.bitwise_not(inverted)
        return inverted

    @staticmethod
    def find_rough_crop_region(inverted: np.ndarray) -> np.ndarray:
        dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        dilated = cv.dilate(inverted, dilation_kernel)

        max_area = 0
        max_index = 0

        contours, hierarchy = cv.findContours(dilated, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            a = cv.contourArea(contours[i], oriented=False)
            if a > max_area:
                max_area = a
                max_index = i

        return cv.boundingRect(contours[max_index])

    @staticmethod
    def nms_line(lines: List[Line], min_dist=20.0) -> List[Line]:
        """
        :param lines: Line point pairs, must be sorted according to vote count
        :return:
        """
        out_lines = []
        for i, l1 in enumerate(lines):
            add = True
            for j, l2 in enumerate(lines[i + 1:]):
                if abs(l1[0] - l2[0]) < min_dist:
                    add = False

            if add:
                out_lines.append(l1)

        return out_lines

    @staticmethod
    def find_lines(image: np.ndarray, theta: float) -> Tuple[List[Line], List[Line]]:
        pi2 = math.pi / 2

        dist_resolution = 1
        angle_resolution = math.pi / 360
        support_threshold = 450

        lines = cv.HoughLines(
            image, dist_resolution, angle_resolution, support_threshold, srn=0, stn=0, min_theta=0, max_theta=math.pi)
        h_lines, v_lines = [], []
        for line in lines:
            line = line.flat
            if 0 <= line[0] <= theta or math.pi - theta <= line[1] <= math.pi:
                h_lines.append(line)
            elif pi2 - theta <= line[1] <= pi2 + theta:
                v_lines.append(line)
            else:
                print("Could not match line.")

        return SudokuDetector.nms_line(h_lines), SudokuDetector.nms_line(v_lines)

    @staticmethod
    def intersect(l1: Line, l2: Line, epsilon=1e-15) -> Point:

        n = np.array([np.cos(l1[1]), np.sin(l1[1])])
        n = n / np.linalg.norm(n)

        m = np.array([np.cos(l2[1]), np.sin(l2[1])])
        m = m / np.linalg.norm(m)

        n2m1 = n[1] * m[0]
        n1m2 = n[0] * m[1]

        if abs(n2m1 - n1m2) < epsilon:
            raise ValueError("Parallel lines cannot be intersected")

        x = (l2[0] * n[1] - l1[0] * m[1]) / (n2m1 - n1m2)
        y = (l2[0] * n[0] - l1[0] * m[0]) / (n1m2 - n2m1)

        return x, y

    @staticmethod
    def find_min_max_line_pair(lines: List[Line], angle_function: Callable[[float], float]) -> Tuple[Line, Line]:
        min_line = lines[0]
        max_line = lines[0]

        for line in lines:
            dx = angle_function(line[1] * line[0])
            if dx > angle_function(max_line[1]) * max_line[0]:
                max_line = line
            if dx < angle_function(min_line[1]) * min_line[0]:
                min_line = line

        return min_line, max_line

    def detect_sudoku_corners(self, sudoku: Sudoku, inverted: np.ndarray) -> None:

        dsize: Size = (512, 512)

        box = SudokuDetector.find_rough_crop_region(inverted)
        p1 = (box[1], box[1] + box[3])
        p2 = (box[0], box[0] + box[2])
        sudoku.bbox = (p1, p2)
        crop = cv.resize(inverted[p1[0]:p1[1], p2[0]:p2[1]], dsize)

        h_lines, v_lines = SudokuDetector.find_lines(crop, math.pi / 90)

        if not h_lines or not v_lines:
            raise Exception("Failed to locate enough lines!")

        min_hline, max_hline = SudokuDetector.find_min_max_line_pair(h_lines, math.cos)
        min_vline, max_vline = SudokuDetector.find_min_max_line_pair(v_lines, math.sin)

        p = (sudoku.bbox[0][0], sudoku.bbox[1][0])
        delta = (sudoku.bbox[0][1] / dsize[0], sudoku.bbox[1][1] / dsize[1])

        def uncrop_coords(b: Point) -> Point:
            return p[0] + b[0] * delta[0], p[1] + b[1] * delta[1]

        sudoku.corners = Quad(
            uncrop_coords(SudokuDetector.intersect(min_hline, min_vline)),
            uncrop_coords(SudokuDetector.intersect(max_hline, min_vline)),
            uncrop_coords(SudokuDetector.intersect(max_hline, max_vline)),
            uncrop_coords(SudokuDetector.intersect(min_hline, max_vline)),
        )

    def compute_unwarp_transform(self, sudoku: Sudoku):
        destCorners = Quad(
            (0, 0),
            (self.warp_size[0], 0),
            (self.warp_size[0], self.warp_size[1]),
            (0, self.warp_size[1]),
        )
        sudoku.warp_map = cv.getPerspectiveTransform(sudoku.corners.as_array(), destCorners.as_array())
        sudoku.unwarp_map = cv.getPerspectiveTransform(destCorners.as_array(), sudoku.corners.as_array())
        sudoku.aligned = cv.warpPerspective(sudoku.input, sudoku.warp_map, self.warp_size)

    def get_cells(self, sudoku: Sudoku):
        sudoku.cells = []
        for row in range(9):
            for col in range(9):
                cell_rect = (
                    self.div_size + col * self.pcell_size,
                    self.div_size + row * self.pcell_size,
                    *self.cell_size
                )
                sudoku.cells.append(cell_rect)

    @staticmethod
    def keep_largest_blob(in_mat):
        maximum = -1
        tmp = cv.bitwise_not(in_mat)

        for row in range(tmp.shape[0]):
            for col in range(tmp.shape[1]):
                if tmp[row, col] < 128:
                    continue  # Skip processed and background pixels

                h, w = tmp.shape
                mask = np.zeros((h + 2, w + 2), np.uint8)
                area = cv.floodFill(tmp, mask, (col, row), 64)[0]
                if area <= maximum:
                    continue  # Keep only the largest blob
                maximum = area

        tmp = 128 + in_mat * 0.5
        _, tmp = cv.threshold(tmp, 64, 255, cv.THRESH_BINARY)

        return in_mat if maximum == -1 else tmp

    def get_cell_contents(self, sudoku):
        img = cv.GaussianBlur(sudoku.aligned, (5, 5), 1)
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        img = cv.medianBlur(img, 3)

        sudoku.cell_contents = []
        for cell in sudoku.cells:
            t_cell = img[cell[0]:cell[0] + cell[2], cell[1]:cell[1] + cell[3]]

            # t = 4
            # nnz_thresh = 20
            # s = self.cell_size * self.cell_size
            # for i in range(t):
            # TODO: implement border clipping

            out = SudokuDetector.keep_largest_blob(t_cell)
            output_cell = cv.resize(out, (20, 20))
            output_cell = 255 - output_cell
            sudoku.cell_contents.append(output_cell)


def main():
    pass


if __name__ == '__main__':
    main()
