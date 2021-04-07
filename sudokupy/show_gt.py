from utils import read_ground_truth
import numpy as np
import cv2 as cv

annot = read_ground_truth(np.os.path.abspath('ground_truth_new.csv'))

for file_name, coords in annot:
    coords = np.array(coords).reshape(4, 2)
    img = cv.imread(file_name, cv.IMREAD_COLOR)

    print(f'{file_name}, Shape: {img.shape}')

    img = cv.polylines(img, [coords], True, (0, 255, 0), thickness=5)

    h, w = img.shape[:2]
    cx = int(round(w / 2))
    cy = int(round(h / 2))

    img = cv.drawMarker(img, (cx, cy), (0, 0, 255), thickness=5)

    img = cv.resize(img, (1024, int(img.shape[0] / (img.shape[1] / 1024))))

    cv.imshow('Sudoku Groundtruth', img)
    cv.waitKey()
