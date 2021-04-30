//
// Created by jannik on 13.04.2021.
//

#ifndef SUDOKUDETECTOR_H
#define SUDOKUDETECTOR_H
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

class SudokuDetection
{};

class SudokuDetector
{
    const int scaled_side_len = 1024;

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
    std::vector<cv::Point> padContour(const std::vector<cv::Point>& contour, int padding);
    [[nodiscard]] cv::Mat unwarpPatch(const cv::Mat& image,
                                      const std::array<cv::Point2f, 4>& corners,
                                      const cv::Size& outSize) const;
    std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Point>>> extractCells(const cv::Mat& image);
    [[nodiscard]] cv::Mat binarizeSudoku(const cv::Mat& image) const;
    int classify(cv::Mat cellPatch);
};

#endif // SUDOKUDETECTOR_H
