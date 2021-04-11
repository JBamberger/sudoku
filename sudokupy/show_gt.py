import cv2 as cv

import config
from utils import rotation_correction, read_ground_truth

if __name__ == '__main__':
    for file_name, coords in read_ground_truth(config.sudokus_gt_path):
        img = cv.imread(file_name, cv.IMREAD_COLOR)

        img, coords = rotation_correction(img, coords)

        print(f'{file_name}, Shape: {img.shape}')

        img = cv.polylines(img, [coords], True, (0, 255, 0), thickness=5)

        cv.line(img, tuple(coords[0, :]), tuple(coords[1, :]), (255, 0, 255), thickness=10)
        cv.line(img, tuple(coords[1, :]), tuple(coords[2, :]), (255, 255, 0), thickness=10)

        h, w = img.shape[:2]
        cx = int(round(w / 2))
        cy = int(round(h / 2))

        img = cv.drawMarker(img, (cx, cy), (0, 0, 255), thickness=5)

        img = cv.resize(img, (1024, int(img.shape[0] / (img.shape[1] / 1024))))

        cv.imshow('Sudoku Groundtruth', img)
        cv.waitKey()
