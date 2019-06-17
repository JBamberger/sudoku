//
// Created by Jannik on 28.05.2019.
//

#ifndef SUDOKU_SUDOKUDETECTOR_H
#define SUDOKU_SUDOKUDETECTOR_H


#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

typedef struct {
    cv::Vec2f top = cv::Vec2f(1000, 1000);
    cv::Vec2f bottom = cv::Vec2f(-1000, -1000);
    cv::Vec2f left = cv::Vec2f(1000, 1000);
    cv::Vec2f right = cv::Vec2f(-1000, -1000);
} SudokuEdges;

enum class DetectorType {
    SIMPLE_DETECTOR, LINE_BASED_DETECTOR
};

class SudokuDetector {

public:
    virtual std::vector<cv::Mat> detectDigits(cv::Mat &image) = 0;
    static SudokuDetector *createInstance(DetectorType type);
};


class LineBasedDetector : public SudokuDetector {

public:
    std::vector<cv::Mat> detectDigits(cv::Mat &image) override;
};


class SudokuDetectorImpl : public SudokuDetector {
public:
    std::vector<cv::Mat> detectDigits(cv::Mat &image) override;

private:
    void mergeRelatedLines(std::vector<cv::Vec2f> &lines, cv::Mat &image);
    void find_sudoku_blob(cv::Mat &image, cv::Mat &outerBox);
    SudokuEdges find_sudoku_edges(std::vector<cv::Vec2f> &lines);
    int find_sudoku_corners(cv::Size2i imgSize, SudokuEdges &edges, cv::Point2f(&src)[4], cv::Point2f(&dst)[4]);
};


#endif //SUDOKU_SUDOKUDETECTOR_H
