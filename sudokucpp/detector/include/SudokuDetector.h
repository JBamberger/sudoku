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
    const int scaled_side_len = 1024;
    CellClassifier cellClassifier;

  public:
    std::unique_ptr<SudokuDetection> detect(const cv::Mat& sudokuImage);

  private:
    std::tuple<cv::Mat, double> inputResize(const cv::Mat& sudokuImage);
    std::vector<cv::Point> detectSudoku(const cv::Mat& normSudoku);
    std::vector<cv::Point> locateSudokuContour(const std::vector<std::vector<cv::Point>>& contours,
                                               const cv::Mat& normSudoku);
    void approximateQuad(const std::vector<cv::Point>& contour,
                         std::vector<cv::Point>& outRect,
                         bool normalizeOrientation = true);
    void normalizeQuadOrientation(const std::vector<cv::Point>& contour, std::vector<cv::Point>& outRect);
    static cv::Mat getUnwarpTransform(const std::array<cv::Point2f, 4>& corners, const cv::Size& outSize);
    [[nodiscard]] static cv::Mat unwarpPatch(const cv::Mat& image,
                                             const std::array<cv::Point2f, 4>& corners,
                                             const cv::Size& outSize);
    std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Point>>> extractCells(const cv::Mat& image);
    [[nodiscard]] cv::Mat binarizeSudoku(const cv::Mat& image) const;
};

#endif // SUDOKUDETECTOR_H
