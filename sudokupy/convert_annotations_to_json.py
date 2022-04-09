import json
import os
import pathlib as pl

import cv2 as cv
import numpy as np

from sudokupy.utils import rot, rot_cw, get_project_root


def read_annotations(annot_file):
    with open(annot_file, 'r', encoding='utf8') as f:
        annotations = []
        for line in f:
            fields = tuple(map(str.strip, line.split(',')))
            file = fields[0].strip('"')
            numbers = fields[1].strip('""')
            coords = [int(n) for n in fields[2:10]]
            annotations.append({
                'file': file,
                'numbers': numbers if numbers else None,
                'coordinates': coords
            })
    return annotations


def correct_rotation(annots):
    for annot in annots:
        file_name = files_dir / annot['file']
        coords = annot['coordinates']
        numbers = annot['numbers']

        coords = np.array(coords, dtype=np.int32).reshape(-1, 2)

        orig_img = cv.imread(file_name, cv.IMREAD_COLOR)

        # img, coords = rotation_correction(img, coords)

        print(f'{file_name}, Shape: {orig_img.shape}')

        while True:
            img = orig_img.copy()
            img = cv.polylines(img, [coords], True, (0, 255, 0), thickness=5)

            cv.line(img, tuple(coords[0, :]), tuple(coords[1, :]), (255, 0, 255), thickness=10)
            cv.line(img, tuple(coords[1, :]), tuple(coords[2, :]), (255, 255, 0), thickness=10)

            h, w = img.shape[:2]
            cx = int(round(w / 2))
            cy = int(round(h / 2))

            img = cv.drawMarker(img, (cx, cy), (0, 0, 255), thickness=5)

            ul = coords[0, :]
            ur = coords[1, :]
            lr = coords[2, :]
            ll = coords[3, :]
            for dy in np.linspace(0, 1, 10):
                for dx in np.linspace(0, 1, 10):
                    yl = dy * ul + (1 - dy) * ll
                    yr = dy * ur + (1 - dy) * lr

                    p = dx * yl + (1 - dx) * yr
                    cv.drawMarker(img, tuple(p.astype(np.int32)), (0, 255, 0), cv.MARKER_CROSS, thickness=2)

            img = cv.resize(img, (1024, int(img.shape[0] / (img.shape[1] / 1024))))

            cv.imshow('Sudoku Groundtruth', img)
            k = cv.waitKey()

            if k == ord('q'):
                exit()
            elif k == ord('r'):
                coords = rot(coords, h=w, w=h)
                print(', '.join(map(str, coords.flatten())))
            elif k == ord('l'):
                coords = rot_cw(coords, h=w, w=h)
                print(', '.join(map(str, coords.flatten())))
            else:
                break


def check_integrity(annots):
    print('*' * 100)
    files = {annot['file'] for annot in annots}
    for annot in annots:
        # print(annot)
        if not (files_dir / annot['file']).is_file():
            print(annot['file'])

    print('*' * 100)

    for f in os.listdir(files_dir):
        if f not in files:
            print(f)


def export_json(annots, name='annotations.json'):
    with open(annot_dir / name, 'w', encoding='utf8') as f:
        json.dump(annots, f)

def export_bounds_csv(annots, name='sudoku_bounds.csv'):
    import csv

    with open(annot_dir / name, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        for annot in annots:
            file = files_dir / annot['file']
            coords = annot['coordinates']

            fp = file.relative_to(project_root)
            writer.writerow([fp] + coords)

if __name__ == '__main__':
    project_root = pl.Path(get_project_root())
    data_dir = project_root / 'data'
    sudoku_dir = data_dir / 'sudokus'

    annot_dir = sudoku_dir / 'annotations'
    files_dir = sudoku_dir / 'empty_sudokus'

    annots = read_annotations(annot_dir / 'empty_annotations.txt')
    check_integrity(annots)
    export_json(annots)
    export_bounds_csv(annots)

    # correct_rotation(annots)
