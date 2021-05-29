#pragma once

#include <memory>
#include <opencv2/core/core.hpp>
#include <utility>
#include <vector>

struct Quad
{
    cv::Point2f tl;
    cv::Point2f tr;
    cv::Point2f br;
    cv::Point2f bl;

    [[nodiscard]] inline std::vector<cv::Point2f> asVec() const
    {
        std::vector<cv::Point2f> out;
        out.push_back(tl);
        out.push_back(tr);
        out.push_back(br);
        out.push_back(bl);
        return out;
    }
};

struct Sudoku
{
    // input image
    const cv::Mat input;

    // aligned (square) version of the sudoku
    cv::Mat aligned;

    // bounding box of the first detection
    cv::Rect bbox;

    // corner points in the original image
    Quad corners;

    // sudoku cells in row-major order
    std::vector<cv::Rect2i> cells;

    // sudoku cell contents
    std::vector<cv::Mat> cell_contents;

    // mapping from unwarped to warped
    cv::Mat warpMap;

    // mapping from warped to unwarped
    cv::Mat unwarpMap;

    explicit Sudoku(cv::Mat input)
      : input(std::move(input))
    {}
};

class SudokuDetector
{
    class Impl;
    std::unique_ptr<Impl> pimpl;

  public:
    explicit SudokuDetector();
    ~SudokuDetector();
    Sudoku detect_sudoku(const cv::Mat& input);
};
