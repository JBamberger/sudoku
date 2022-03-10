//
// Created by jannik on 13.04.2021.
//

#ifndef SUDOKUDETECTOR_H
#define SUDOKUDETECTOR_H

#include <CellClassifier.h>
#include <SudokuDetection.h>

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

class SudokuDetector
{
    struct Impl;
    std::unique_ptr<Impl> pimpl;

  public:
    explicit SudokuDetector(const std::string& classifierPath);
    SudokuDetector(SudokuDetector&&) noexcept;
    SudokuDetector(const SudokuDetector&) = delete;
    ~SudokuDetector();
    SudokuDetector& operator=(SudokuDetector&&) noexcept;
    SudokuDetector& operator=(const SudokuDetector&) = delete;

    std::unique_ptr<SudokuDetection> detect(const cv::Mat& sudokuImage);
};



// V2
// binarize with various thresholds, and try for each one
// detect contours
// Recursively detect sudoku in contours:
//      Crop contour
//      find lines
//      cluster lines by direction/angle
//      find clusters with many lines perpendicular to each other

#endif // SUDOKUDETECTOR_H
