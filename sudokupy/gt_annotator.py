import torch
import cv2 as cv
import numpy as np
import os

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


def annotate_bounding_poly():
    with open('ground_truth.txt', 'w', encoding='utf8') as f:
        for i, file in enumerate(sudoku_files):
            print(f'[{i + 1}/{len(sudoku_files)}] {file}')

            line = annotate_sudoku_file(file)
            print(line, end='')
            f.write(line)


def annotate_sudoku_file(file):
    sudoku_img_ori = cv.imread(file, cv.IMREAD_COLOR)
    sudoku_img = scale_down(sudoku_img_ori)
    # roi = cv.selectROI("Image", sudoku_img, showCrosshair=True, fromCenter=False)
    selector = PolygonSelector("Image", sudoku_img, 4)
    roi = selector.loop()

    roi = list(map(lambda p: scale_up(sudoku_img_ori, sudoku_img, p), roi))
    print(roi)

    cells = [file] + [str(a) for point in roi for a in point]

    line = ', '.join(cells) + '\n'
    return line


def fix_annotations():
    gt_file = os.path.abspath('ground_truth.csv')
    annotations = read_ground_truth(gt_file)

    with open('ground_truth_new.csv', 'w', encoding='utf8') as f:
        for file_path, coords in annotations:
            sudoku_ori = cv.imread(file_path, cv.IMREAD_COLOR)

            sudoku = torch.from_numpy(sudoku_ori).permute((2, 0, 1)).unsqueeze(0).float()

            _, _, H_in, W_in = sudoku.shape

            H_out = 256
            W_out = 256
            # [N, H_out, W_out, 2]
            grid = torch.zeros((1, H_out, W_out, 2))

            ul = coords[0:2]
            ur = coords[2:4]
            br = coords[4:6]
            bl = coords[6:8]

            large = sudoku_ori.copy()
            small = scale_down(large)

            polyline = np.array([ul, ur, br, bl]).astype(np.float32)
            polyline2 = polyline / (large.shape[0] / small.shape[0])

            polyline[:, 0] = polyline2[:, 0] * large.shape[1] / small.shape[1]
            polyline[:, 1] = polyline2[:, 1] * large.shape[0] / small.shape[0]

            polyline = polyline.round().astype(np.int32)

            cells = [file_path] + [str(a) for a in polyline.flatten()]

            line = ', '.join(cells) + '\n'
            print(line, end='')
            f.write(line)


if __name__ == '__main__':
    annotate_bounding_poly()
    fix_annotations()
