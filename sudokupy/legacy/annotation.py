from __future__ import annotations

from abc import ABC
from typing import Tuple, Callable, NamedTuple, List, Optional

import cv2 as cv
import numpy as np


class Size(NamedTuple):
    width: int
    height: int


class Point(NamedTuple):
    x: int
    y: int


class Rect(NamedTuple):
    p1: Point
    p2: Point


Image = np.ndarray
Polygon = List[Point]


def draw_polygon(img: Image, polygon: Polygon, closed: bool, color: Tuple[int, int, int], *args, **kwargs):
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    cv.polylines(img, [pts], closed, color, *args, **kwargs)


class ScaleTransform:
    def __init__(self, scale_x: float, scale_y: float):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, p: Point) -> Point:
        return Point(
            x=int(p.x * self.scale_x),
            y=int(p.y * self.scale_y),
        )


def scale_image(img: Image, size: Size) -> Tuple[Image, Callable[[Point], Point], Callable[[Point], Point]]:
    scale_x = size.width / img.shape[1]
    scale_y = size.height / img.shape[0]
    scale = min(scale_x, scale_y)

    new_size = Size(width=int(img.shape[1] * scale), height=int(img.shape[0] * scale))
    resized_img = cv.resize(img, new_size, interpolation=cv.INTER_CUBIC)

    return resized_img, ScaleTransform(scale, scale), ScaleTransform(1 / scale, 1 / scale)


class InteractionMode(ABC):
    annotator: Annotator

    def __init__(self, annotator: Annotator):
        self.annotator = annotator

    def draw(self, canvas: Image):
        pass

    def on_mouse(self, event, pos: Point):
        pass

    def on_key(self, key):
        pass


class RectSelectionMode(InteractionMode):
    marker_point: Optional[Point] = None
    cursor_pos: Optional[Point] = None

    def __init__(self, annotator: Annotator):
        super().__init__(annotator)

    def draw(self, canvas: Image):
        if self.marker_point is not None:
            cv.rectangle(canvas, self.marker_point, self.cursor_pos, (255, 0, 0))

    def on_mouse(self, event, pos: Point):
        if event == cv.EVENT_LBUTTONDOWN:
            self.marker_point = pos
        elif event == cv.EVENT_LBUTTONUP:
            if self.marker_point is not None:
                self.annotator.selections.append(Rect(self.marker_point, pos))
            self.marker_point = None
        elif event == cv.EVENT_RBUTTONDOWN:
            self.marker_point = None
        elif event == cv.EVENT_MOUSEMOVE:
            self.cursor_pos = pos


class PolySelectionMode(InteractionMode):
    points: Polygon
    cursor_pos: Point = None

    def __init__(self, annotator: Annotator):
        super().__init__(annotator)
        self.points = []

    def draw(self, canvas: Image):
        for point in self.points:
            cv.drawMarker(canvas, point, (255, 0, 0))
        if len(self.points) > 0 and self.cursor_pos is not None:
            cv.line(canvas, self.points[-1], self.cursor_pos, (255, 255, 0))
        if len(self.points) >= 2:
            draw_polygon(canvas, self.points, False, (255, 0, 0))

        super().draw(canvas)

    def on_mouse(self, event, pos: Point):
        if event == cv.EVENT_LBUTTONUP:
            self.points.append(pos)
        elif event == cv.EVENT_MOUSEMOVE:
            self.cursor_pos = pos

    def on_key(self, key):
        if key == ord('q'):
            self.points = []
        elif key == ord('w'):
            if len(self.points) >= 3:
                self.annotator.polygons.append(self.points)
            self.points = []


class EditMode(InteractionMode):
    index: int

    def __init__(self, annotator: Annotator):
        super().__init__(annotator)
        self.index = 0

    def select_next(self):
        length = len(self.annotator.selections)
        self.index = 0 if length == 0 else (self.index + 1) % length

    def select_prev(self):
        length = len(self.annotator.selections)
        self.index = 0 if length == 0 else (self.index - 1) % length

    def delete_selection(self):
        if 0 <= self.index < len(self.annotator.selections):
            self.annotator.selections.pop(self.index)
            if self.index > 0:
                self.index -= 1

    def draw(self, canvas: Image):
        if 0 <= self.index < len(self.annotator.selections):
            sel = self.annotator.selections[self.index]
            cv.rectangle(canvas, sel.p1, sel.p2, (255, 255, 0))

    def on_key(self, key):
        if key == ord('n') & 0xff:
            self.select_next()
        elif key == ord('m') & 0xff:
            self.select_prev()
        elif key == ord('d') & 0xff:
            self.delete_selection()
        else:
            print(f'KeyPress: {key}')


class Annotator:
    selections: List[Rect] = []
    polygons: List[Polygon] = []
    mode: InteractionMode

    def __init__(self, path):
        img = cv.imread(path)
        self.img, self.forward_transform, self.backward_transform = scale_image(img, Size(width=1920, height=1080))
        cv.imshow('image', self.img)

        self.mode = RectSelectionMode(self)

    def render(self):
        canvas = self.img.copy()

        for selection in self.selections:
            cv.rectangle(canvas, selection.p1, selection.p2, (255, 0, 255))

        for poly in self.polygons:
            draw_polygon(canvas, poly, True, (255, 0, 255))

        self.mode.draw(canvas)

        cv.imshow('image', canvas)

    def start(self):
        def on_mouse(event, x, y, flags, param):
            self.mode.on_mouse(event, Point(x=x, y=y))
            self.render()

        cv.setMouseCallback("image", on_mouse)

        self.loop()

    def loop(self):
        while True:
            key = cv.waitKey(0)
            if key == -1:
                break
            elif key == ord('e') & 0xff:
                self.mode = EditMode(self)
            elif key == ord('s') & 0xff:
                self.mode = RectSelectionMode(self)
            elif key == ord('p') & 0xff:
                self.mode = PolySelectionMode(self)
            else:
                self.mode.on_key(key)
            self.render()

        cv.destroyAllWindows()


if __name__ == '__main__':
    annotator = Annotator("E:/pictures/others/_needs_sorting/sudoku/only1/sudoku-sk.jpg")
    annotator.start()
