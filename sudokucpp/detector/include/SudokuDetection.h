//
// Created by jannik on 18.05.2021.
//

#ifndef SUDOKUDETECTION_H
#define SUDOKUDETECTION_H

#include <array>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <utility>
#include <vector>

struct SudokuDetection
{
    const double inputScale;

    std::array<cv::Point2f, 4> sudokuCorners;
    cv::Mat unwarpTransform;
    bool foundSudoku = false;

    std::array<std::vector<cv::Point2f>, 81> cells;
    std::array<int, 81> cellLabels;
    bool foundAllCells = false;

    explicit SudokuDetection(double inputScale);
    ~SudokuDetection();

    void drawSudokuBounds(cv::Mat& canvas) const;
    void drawCells(cv::Mat& canvas) const;
    void drawLabels(cv::Mat& canvas, bool drawEmpty) const;

    cv::Mat renderOverlay() const;
};

#endif // SUDOKUDETECTION_H
