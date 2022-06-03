//
// Created by jannik on 25.08.2021.
//

#ifndef SUDOKU_APP_DETECT_SUDOKU_H
#define SUDOKU_APP_DETECT_SUDOKU_H

#include "geometry.h"
#include <opencv2/core.hpp>
#include <vector>

cv::Mat
threshMorphed(const cv::Mat& image, int morphSize = 25);

Contour
locateSudokuContour(const std::vector<Contour>& contours, const cv::Point2f& imageCenter);

bool
detectSudoku(const cv::Mat& sudokuImage, std::array<cv::Point2f, 4>& sudokuCorners, int scaled_side_len);

std::vector<Contour>
findSquares(const cv::Mat& image);

cv::Mat
binarizeSudoku(const cv::Mat& grayImage);

void
detectCellsFast(size_t height, size_t width, std::array<Contour, 81>& cellCoords);

void
detectCells(const cv::Mat& grayImage, std::array<Contour, 81>& cellCoords);

void
detectCellsRobust(const cv::Mat& grayImage, std::array<Contour, 81>& cellCoords);

void
fillMissingSquares(std::array<Contour, 81>& cellCoordinates);

#endif // SUDOKU_APP_DETECT_SUDOKU_H
