import os

import cv2 as cv
import numpy as np

from sudoku_detector import SudokuDetector, Point
from sudoku_util import Sudoku

source = 'E:\pictures\others\_needs_sorting\sudoku\only1'


def is_image(file):
    img_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    return os.path.isfile(file) and os.path.splitext(file)[-1].lower() in img_extensions


def main():
    # sudoku_files = filter(is_image, [os.path.join(source, file) for file in os.listdir(source)])
    #
    # for f in sudoku_files:
    #     img = cv.imread(f)
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #     plt.imshow(img)
    #     plt.show()

    img = cv.imread("E:\pictures\others\_needs_sorting\sudoku\only1\sudoku-sk.jpg", cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, dsize=None, fx=0.5, fy=0.5)
    sudoku: Sudoku = SudokuDetector().detect_sudoku(img)

    result_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    def unwarp(p: Point):
        warp_p = (p[0], p[1], 1.0)
        u = sudoku.unwarp_map * warp_p
        u = u / u[2, 0]
        return int(u[0, 0]), int(u[1, 0])

    # for cell in sudoku.cells:
    #     pts = [
    #         unwarp((cell[0], cell[1])),
    #         unwarp((cell[0] + cell[2], cell[1])),
    #         unwarp((cell[0] + cell[2], cell[1] + cell[3])),
    #         unwarp((cell[0], cell[1] + cell[3])),
    #     ]
    #     cv.polylines(result_img, np.array(pts), True, (255, 0, 0))

    c = sudoku.corners
    pts = np.array((c.tl, c.tr, c.br, c.bl), dtype=np.int32)
    # cv.polylines(result_img, pts, True, (0, 255, 0))
    cv.rectangle(result_img, *sudoku.bbox, (0, 0, 255))
    cv.imshow("Result", result_img)
    cv.waitKey()


if __name__ == '__main__':
    main()
