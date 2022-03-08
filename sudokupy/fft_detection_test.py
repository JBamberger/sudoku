import time

import cv2 as cv
import numpy as np

import config
from utils import read_ground_truth, show

import matplotlib.pyplot as plt

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)


def detect(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_float = image_gray.astype(np.float32)
    image_float = image_float / 255.0

    m2 = cv.GaussianBlur(image_float, (7,7), 1)

    delta = 1
    N = m2.shape[0]
    fy = 1 / (N * delta) * (np.arange(N) - 1 - N / 2)
    M = m2.shape[1]
    fx = 1 / (M * delta) * (np.arange(M) - 1 - M / 2)


    fft_vecx = np.fft.fftshift(np.fft.fft(m2.sum(axis=0)))
    fft_vecy = np.fft.fftshift(np.fft.fft(m2.T.sum(axis=0)))

    plt.plot(fx, np.abs(fft_vecx))
    plt.plot(fy, np.abs(fft_vecy))
    plt.ylim(-10, 10000)
    plt.show()







    return image


if __name__ == '__main__':
    gt_annoatations = read_ground_truth(config.sudokus_gt_path)
    for sudoku_index, (file_path, gt_coords) in enumerate(gt_annoatations):
        if sudoku_index < 0:
            continue

        start = time.time()
        sudoku = cv.imread(file_path)

        canvas = detect(sudoku)

        print('Took ', time.time() - start)
        show(canvas)
        key = cv.waitKey()
        if key == ord('q'):
            exit()
