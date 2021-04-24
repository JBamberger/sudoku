//
// Created by jannik on 13.04.2021.
//

#include "SudokuDetector.h"
#include <algorithm>
#include <array>
#include <opencv2/imgproc.hpp>
//#include <torch/torch.h>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <utils.h>

void
show(const cv::Mat& mat)
{
    cv::imshow("Mat", mat);
    cv::waitKey();
}

void
show(const std::string& title, const cv::Mat& mat)
{
    cv::imshow(title, mat);
    cv::waitKey();
}

template<typename T>
T
orientedAngle(const T& x1, const T& y1, const T& x2, const T& y2)
{
    auto dot = x1 * x2 + y1 * y2;
    auto det = x1 * y2 - y1 * x2;
    auto angle = std::atan2(det, dot);
    return angle;
}

std::unique_ptr<SudokuDetection>
SudokuDetector::detect(cv::Mat sudokuImage)
{

    auto [normScaleSudoku, inputDownscale] = inputResize(sudokuImage);

    auto detection = std::make_unique<SudokuDetection>();

    auto sudokuLocation = detectSudoku(normScaleSudoku);
    std::cout << sudokuLocation << std::endl;

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

    return std::make_tuple(scaledImg, scale);
}
std::vector<cv::Point>
SudokuDetector::detectSudoku(cv::Mat normSudoku)
{
    cv::Mat sudokuGray;
    cv::cvtColor(normSudoku, sudokuGray, cv::COLOR_BGR2GRAY);

    cv::Mat closingKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
    cv::Mat closing;
    cv::morphologyEx(sudokuGray, closing, cv::MORPH_CLOSE, closingKernel);

    sudokuGray = (sudokuGray / closing * 255); // TODO: requires casting.
                                               //    cv::Mat tempMat;
                                               //    sudokuGray.convertTo(tempMat, CV_32F);
                                               //    tempMat = (tempMat / closing * 255); // TODO: requires casting.
                                               //    tempMat.convertTo(sudokuGray, CV_8U);

    cv::Mat sudokuBin;
    // TODO: might require | or  & instead of +
    cv::threshold(sudokuGray, sudokuBin, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    cv::Mat dilationKernel;
    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat dilated;
    cv::dilate(sudokuBin, dilated, dilationKernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    auto points = locateSudokuContour(contours, normSudoku);

    cv::Mat canvas;
    cv::cvtColor(sudokuGray, canvas, cv::COLOR_GRAY2BGR);


    std::cout << points << std::endl;
    std::vector<std::vector<cv::Point>> lines;
    lines.push_back(points);
    cv::polylines(canvas, lines, true, cv::Scalar(0,255,0));
    cv::imshow("Bounds", canvas);
    cv::waitKey();

    return points;
}
std::vector<cv::Point>
SudokuDetector::locateSudokuContour(const std::vector<std::vector<cv::Point>>& contours, cv::Mat normSudoku)
{

    double maxArea = 0.0;
    int maxIndex = -1;
    cv::Point2f imageCenter(static_cast<float>(normSudoku.cols) / 2.f, static_cast<float>(normSudoku.rows) / 2.f);
    for (auto i = 0; i < contours.size(); i++) {
        const auto& contour = contours.at(i);

        auto polytest = cv::pointPolygonTest(contour, imageCenter, false);
        if (polytest < 0)
            continue;

        std::array<cv::Point2f, 4> points;
        cv::minAreaRect(contour).points(points.data());

        auto d1 = cv::norm(points.at(0) - points.at(1));
        auto d2 = cv::norm(points.at(0) - points.at(2));

        auto squareThresh = (d1 + d2) / 2.0 * 0.5;
        if (abs(d1 - d2) > squareThresh)
            continue;

        auto contourArea = cv::contourArea(contour, false);
        if (contourArea > maxArea) {
            maxArea = contourArea;
            maxIndex = i;
        }
    }

    std::vector<cv::Point> outputPoints;

    if (maxIndex >= 0) {
        const auto& bestContour = contours.at(maxIndex);

        approximateQuad(bestContour, outputPoints, true);
    }

    return outputPoints;
}
void
SudokuDetector::approximateQuad(const std::vector<cv::Point>& contour,
                                std::vector<cv::Point>& outRect,
                                bool normalizeOrientation)
{
    auto n = contour.size();

    double maxDist = 0.0;

    std::array<size_t, 4> pointIndices{ 0, 0, 0, 0 };

    std::vector<std::vector<double>> distances;
    distances.reserve(n);
    for (int i = 0; i < n; i++) {
        distances.emplace_back(n);
        for (int j = 0; j < n; j++) {
            double dist = cv::norm(contour.at(i) - contour.at(j));
            distances.at(i).at(j) = dist;

            if (dist > maxDist) {
                maxDist = dist;
                pointIndices[0] = i;
                pointIndices[1] = j;
            }
        }
    }

    maxDist = 0;
    std::vector<double> distSum;
    distSum.reserve(n);
    for (int i = 0; i < n; i++) {
        auto s = distances.at(pointIndices[0]).at(i) + distances.at(pointIndices[1]).at(i);
        distSum.push_back(s);
        if (s > maxDist) {
            maxDist = s;
            pointIndices[2] = i;
        }
    }
    maxDist = 0;
    for (int i = 0; i < n; i++) {
        auto s = distSum.at(i) + distances.at(pointIndices[2]).at(i);
        if (s > maxDist) {
            maxDist = s;
            pointIndices[3] = i;
        }
    }

    std::sort(std::begin(pointIndices), std::end(pointIndices));

    outRect.clear();
    for (int i = 0; i < 4; i++) {
        outRect.push_back(contour.at(pointIndices[i]));
    }

    if (normalizeOrientation) {
        normalizeQuadOrientation(outRect, outRect);
    }
}
void
SudokuDetector::normalizeQuadOrientation(const std::vector<cv::Point>& contour, std::vector<cv::Point>& outRect)
{
    auto sortX = argsort<cv::Point>(contour, [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
    auto sortY = argsort<cv::Point>(contour, [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });

    std::array<size_t, 4> bins{ 0, 0, 0, 0 };
    bins[sortX[0]] += 1;
    bins[sortX[1]] += 1;
    bins[sortY[0]] += 1;
    bins[sortY[1]] += 1;

    int offset = 0;
    for (int i = 0; i < 4; i++) {
        if (bins[i] >= 2) {
            offset = i;
            break;
        }
    }
    std::vector<cv::Point> rotatedBox;
    rotatedBox.reserve(contour.size());
    for (int i = 0; i < contour.size(); i++) {
        auto index = i - offset;
        if (index < 0) {
            index += static_cast<int>(contour.size());
        }
        rotatedBox.push_back(contour.at(index));
    }

    auto delta = rotatedBox.at(1) - rotatedBox.at(0);
    auto angle = orientedAngle<double>(static_cast<double>(delta.x), static_cast<double>(delta.y), 1.0, 0.0);

    outRect.at(0) = rotatedBox.at(0);
    if (angle < -45.0 || 45.0 < angle) {
        for (int i = rotatedBox.size() - 1; i > 0; i--) {
            outRect.at(i) = rotatedBox.at(i);
        }
    } else {
        for (int i = 1; i < rotatedBox.size(); i++) {
            outRect.at(i) = rotatedBox.at(i);
        }
    }
}
