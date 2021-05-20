//
// Created by jannik on 18.05.2021.
//

#ifndef SUDOKUDETECTION_H
#define SUDOKUDETECTION_H

#include <array>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>

struct SudokuDetection
{
    std::array<cv::Point2f, 4> sudokuCorners;
    cv::Mat unwarpTransform;
    bool foundSudoku = false;

    std::array<std::vector<cv::Point>, 81> cellCoords;
    std::array<int, 81> cellLabels;
    bool foundAllCells = false;

    void drawOverlay(cv::Mat& canvas) const;
};

#endif // SUDOKUDETECTION_H
