from typing import Tuple

import numpy as np


class Quad:
    tl: Tuple[int, int]
    tr: Tuple[int, int]
    br: Tuple[int, int]
    bl: Tuple[int, int]

    def __init__(self, tl, tr, br, bl):
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl

    def as_list(self):
        return [self.tl, self.tr, self.br, self.bl]

    def as_array(self):
        return np.array(self.as_list(), dtype=np.float32)


class Sudoku:
    input: np.ndarray
    aligned: np.ndarray
    bbox: Tuple[Tuple[int, int], Tuple[int, int]]
    corners: Quad
    cells: np.ndarray  # shape: [n,4]
    cell_contents: np.ndarray  # shape: [n,20,20]
    warp_map: np.ndarray
    unwarp_map: np.ndarray

    def __init__(self, input_img):
        self.input = input_img
        self.aligned = None
        self.bbox = None
        self.corners = None
        self.cells = None
        self.cell_contents = None
        self.warp_map = None
        self.unwarp_map = None
