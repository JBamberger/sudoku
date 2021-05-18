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
    explicit SudokuDetector();
    SudokuDetector(SudokuDetector&&) noexcept;
    SudokuDetector(const SudokuDetector&) = delete;
    ~SudokuDetector();
    SudokuDetector& operator=(SudokuDetector&&) noexcept;
    SudokuDetector& operator=(const SudokuDetector&) = delete;

    std::unique_ptr<SudokuDetection> detect(const cv::Mat& sudokuImage);
};

#endif // SUDOKUDETECTOR_H
