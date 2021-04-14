//
// Created by jannik on 13.04.2021.
//

#include "SudokuDetector.h"
#include <opencv2/imgproc.hpp>

std::unique_ptr<SudokuDetection>
SudokuDetector::detect(cv::Mat sudokuImage)
{
    auto detection = std::make_unique<SudokuDetection>();

    return detection;
}
std::tuple<cv::Mat, double>
SudokuDetector::inputResize(cv::Mat sudokuImage, size_t long_side)
{
    int h = sudokuImage.rows;
    int w = sudokuImage.cols;

    double scale = static_cast<double>(long_side) / static_cast<double>(h > w ? h : w);

    cv::Mat scaledImg;
    cv::resize(sudokuImage, scaledImg, cv::Size(), scale, scale, cv::INTER_AREA);

      return std::make_tuple(scale, scaledImg);
}
