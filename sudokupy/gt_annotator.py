import os

import cv2 as cv
import numpy as np

from utils import read_ground_truth

sudoku_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'share', 'only1')
sudoku_files = [os.path.join(sudoku_dir, fn) for fn in os.listdir(sudoku_dir) if os.path.splitext(fn)[-1] == '.jpg']


class PolygonSelector:

    def __init__(self, window_name, image, num_points, max_click_delta=20):
        self.window_name = window_name
        self.image = image
        self.num_points = num_points
        self.points = []
        self.click_delta_squared = max_click_delta * max_click_delta

        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.onMouse)
        self.points_changed()

    def loop(self):
        while True:
            key = cv.waitKey()
            # print(key)
            if key == 0x0D:
                if len(self.points) == self.num_points:
                    return self.points
                else:
                    print('Not enough points selected.')
            elif key == 0x08:
                print('Deleting last point.')
                self.points.pop()
                self.points_changed()
            elif key == ord('r'):
                print('Resetting points.')
                self.points = []
                self.points_changed()

    def add_point(self, pos):
        if len(self.points) == self.num_points:
            print('Max number of points reached.')
        else:
            self.points.append(pos)
        self.points_changed()

    def points_changed(self):
        img = self.image.copy()

        l = len(self.points)
        if l <= 0:
            pass
        elif l == 1:
            pass

        for point in self.points:
            cv.drawMarker(img, point, (0, 255, 0))

        if l >= 2:
            cv.polylines(img, [np.array(self.points)], l == self.num_points, (0, 255, 0))

        cv.imshow(self.window_name, img)

    def dist_squared(self, p1, p2) -> float:
        a = p1[0] - p2[0]
        b = p1[1] - p2[1]
        return a * a + b * b

    def onMouse(self, event: int, x: int, y: int, flags: int, *args):
        # EVENT_FLAG_ALTKEY = 32
        # EVENT_FLAG_CTRLKEY = 8
        # EVENT_FLAG_LBUTTON = 1
        # EVENT_FLAG_MBUTTON = 4
        # EVENT_FLAG_RBUTTON = 2
        # EVENT_FLAG_SHIFTKEY = 16
        #
        # event_map = {
        #     7: 'EVENT_LBUTTONDBLCLK',
        #     1: 'EVENT_LBUTTONDOWN',
        #     4: 'EVENT_LBUTTONUP',
        #     9: 'EVENT_MBUTTONDBLCLK',
        #     3: 'EVENT_MBUTTONDOWN',
        #     6: 'EVENT_MBUTTONUP',
        #     11: 'EVENT_MOUSEHWHEEL',
        #     0: 'EVENT_MOUSEMOVE',
        #     10: 'EVENT_MOUSEWHEEL',
        #     8: 'EVENT_RBUTTONDBLCLK',
        #     2: 'EVENT_RBUTTONDOWN',
        #     5: 'EVENT_RBUTTONUP',
        # }

        position = (x, y)

        if event == cv.EVENT_LBUTTONDOWN:
            self.down_pos = position
        elif event == cv.EVENT_LBUTTONUP:
            # This check also ensures that the click happened within valid coordinates, because a click can only
            # start in the valid area.
            if self.dist_squared(self.down_pos, position) < self.click_delta_squared:
                self.add_point(position)
            else:
                # no click aborted
                pass

        # if event != 0:
        #     print(event_map[event])
        #     print(x, ' ', y)
        #     print(flags)
        #     print(args)


def scale_down(image, size=1024):
    h, w, c = image.shape

    if h > w:
        new_w = int(size / h * w)
        new_size = (size, new_w)
    else:
        new_h = int(size / w * h)
        new_size = (new_h, size)

    return cv.resize(image, new_size, interpolation=cv.INTER_AREA)


def scale_up(large, small, point):
    sf = large.shape[0] / small.shape[0]

    return int(round(point[0] * sf)), int(round(point[1] * sf))


def scale_up_polyline(large, small, points):
    polyline = points.astype(np.float32)

    polyline[:, 0] *= large.shape[1] / small.shape[1]
    polyline[:, 1] *= large.shape[0] / small.shape[0]

    polyline = polyline.round().astype(np.int32)

    return polyline


def annotate_sudoku_file(file):
    sudoku_img_ori = cv.imread(file, cv.IMREAD_COLOR)

    sudoku_img = scale_down(sudoku_img_ori)

    # roi = cv.selectROI("Image", sudoku_img, showCrosshair=True, fromCenter=False)
    selector = PolygonSelector("Image", sudoku_img, 4)
    roi = selector.loop()

    roi = scale_up_polyline(sudoku_img_ori, sudoku_img, np.array(roi).reshape(4, 2))
    print(roi)

    cells = [file] + [str(a) for a in roi.flatten()]

    line = ', '.join(cells) + '\n'
    return line


def annotate_bounding_poly():
    with open('ground_truth.txt', 'w', encoding='utf8') as f:
        for i, file in enumerate(sudoku_files):
            print(f'[{i + 1}/{len(sudoku_files)}] {file}')

            line = annotate_sudoku_file(file)
            print(line, end='')
            f.write(line)


def fix_annotations():
    with open('ground_truth_new.csv', 'w', encoding='utf8') as f:
        for file_path, coords in read_ground_truth(os.path.abspath('ground_truth.csv')):
            sudoku_ori = cv.imread(file_path, cv.IMREAD_COLOR)

            large = sudoku_ori.copy()
            small = scale_down(large)

            scale_x = large.shape[1] / small.shape[1]
            scale_y = large.shape[0] / small.shape[0]

            polyline = coords.astype(np.float32)

            # reverse scale_up function
            polyline = polyline / scale_y

            # perform correct scaling
            polyline[:, 0] *= scale_x
            polyline[:, 1] *= scale_y

            polyline = polyline.round().astype(np.int32)

            cells = [file_path] + [str(a) for a in polyline.flatten()]
            line = ', '.join(cells) + '\n'
            print(line, end='')
            f.write(line)


def rename_sudokus():
    with open('ground_truth.renamed.txt', 'w', encoding='utf8') as f:
        for i, (file_path, coords) in enumerate(read_ground_truth(os.path.abspath('ground_truth.txt'))):
            i += 31
            new_path = os.path.join(os.path.dirname(file_path), f'sudoku_{i:d}.jpg')

            if os.path.exists(new_path):
                raise RuntimeError()

            os.rename(file_path, new_path)

            cells = [new_path] + [str(a) for a in coords.flatten()]
            line = ', '.join(cells) + '\n'
            print(line, end='')
            f.write(line)


if __name__ == '__main__':
    annotate_bounding_poly()
    # fix_annotations()
    # rename_sudokus()
