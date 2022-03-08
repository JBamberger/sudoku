//
// Created by jannik on 25.08.2021.
//

#ifndef SUDOKU_APP_DETECT_SUDOKU_H
#define SUDOKU_APP_DETECT_SUDOKU_H

#include <opencv2/core.hpp>
#include <vector>

using Contour = std::vector<cv::Point>;

cv::Mat
threshMorphed(const cv::Mat& image, int morphSize = 25);

void
normalizeQuadOrientation(const Contour& contour, Contour& outRect);

void
approximateQuad(const Contour& contour, Contour& outRect, bool normalizeOrientation = true);

Contour
locateSudokuContour(const std::vector<Contour>& contours, const cv::Point2f& imageCenter);

bool
detectSudoku(const cv::Mat& sudokuImage, std::array<cv::Point2f, 4>& sudokuCorners, int scaled_side_len);

std::vector<Contour>
convexContourNms(const std::vector<Contour>& contours, double thresh);

double
angleCos(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2);

std::vector<Contour>
findSquares(const cv::Mat& image);

double
contourIoU(const Contour& a, const Contour& b);

cv::Mat
binarizeSudoku(const cv::Mat& image);

void
detectCellsFast(size_t height, size_t width, std::array<Contour, 81>& cellCoords);

void
detectCells(const cv::Mat& image, std::array<Contour, 81>& cellCoords);

void
detectCellsRobust(const cv::Mat& image, std::array<Contour, 81>& cellCoords);

void
fillMissingSquares(std::array<Contour, 81>& cellCoordinates);

#endif // SUDOKU_APP_DETECT_SUDOKU_H
