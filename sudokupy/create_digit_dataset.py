import json
import math
import os
import pathlib as pl

import cv2 as cv
import numpy as np


def main():
    data_root = pl.Path(__file__).parent.parent / "data"

    annotation_file = data_root / 'sudokus' / 'annotations' / 'annotations.json'
    sudoku_image_dir = data_root / 'sudokus' / 'empty_sudokus'

    with open(annotation_file, 'r', encoding='utf8') as f:
        annotations = json.load(f)

    for annotation in annotations:
        sudoku_image_path = sudoku_image_dir / annotation['file']
        sudoku_contents = annotation['numbers']
        sudoku_coords = np.array(annotation['coordinates']).reshape(4, 2).astype(np.float32)

        sudoku_image = cv.imread(str(sudoku_image_path), cv.IMREAD_GRAYSCALE)

        out_size = 1024
        pad = int(out_size * 0.05)
        l, h = 0.0 + pad, out_size - pad
        dest_coords = np.array([[l, l], [h, l], [h, h], [l, h]], dtype=np.float32)
        norm_mat = cv.getPerspectiveTransform(sudoku_coords, dest_coords)
        sudoku_normed = cv.warpPerspective(sudoku_image.astype(np.float32) / 255.0,
                                           norm_mat, (out_size, out_size), flags=cv.INTER_AREA)
        sudoku_normed_u8 = np.uint8(sudoku_normed * 255)

        smooth = cv.GaussianBlur(sudoku_normed_u8, ksize=(11, 11), sigmaX=1)
        median_img = np.clip(cv.medianBlur(smooth, ksize=63), 1, 255)
        sub_med = np.float32(smooth) / np.float32(np.maximum(median_img, smooth))
        print(sub_med.min(), sub_med.max(), smooth.min(), smooth.max(), median_img.min(), median_img.max())

        sm_int = (sub_med * 255).astype(np.uint8)

        thresh, sub_med_bin = cv.threshold(sm_int, 200, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        line_canvas = cv.cvtColor(sudoku_normed, cv.COLOR_GRAY2BGR)
        lines = cv.HoughLinesP(255 - sub_med_bin, rho=1.0, theta=np.pi / 180,
                               threshold=127, minLineLength=30, maxLineGap=10)
        for line in lines[:, 0, :]:
            cv.line(line_canvas, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, 8)

        # import matplotlib.pyplot as plt
        # plt.hist(sm_int.flatten(), bins=255)
        # plt.vlines(thresh, 0, 100_000, colors='g')
        # plt.title(thresh)
        # plt.show()

        cv.imshow('finite', np.uint8(np.isfinite(sub_med) * 255))
        cv.imshow('med_img', median_img)
        cv.imshow('sub_med', sub_med)
        cv.imshow('sub_med_bin', sub_med_bin)

        cv.imshow('Sudoku', sudoku_image)
        cv.imshow('Normed', sudoku_normed)
        cv.imshow('HoughLines', line_canvas)
        key = cv.waitKey()
        if key == ord('n'):
            continue
        elif key == ord('q'):
            break
        else:
            pass


if __name__ == '__main__':
    main()
