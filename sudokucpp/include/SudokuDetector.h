//
// Created by jannik on 13.04.2021.
//

#ifndef SUDOKUDETECTOR_H
#define SUDOKUDETECTOR_H
#include <memory>
#include <opencv2/core.hpp>

class SudokuDetection
{};

class SudokuDetector
{

    std::tuple<cv::Mat, double> inputResize(cv::Mat sudokuImage, size_t long_side=1024);
    cv::Mat detectSudoku(cv::Mat normSudoku);
    void padContours(cv::Mat contour, int padding);
    void unwarpPatch(cv::Mat image, cv::Mat corners, bool return_grid = true);
    void extractCells(cv::Mat image);
    void classify(cv::Mat cellPatch);

      public : std::unique_ptr<SudokuDetection> detect(cv::Mat sudokuImage);
};

#endif // SUDOKUDETECTOR_H
