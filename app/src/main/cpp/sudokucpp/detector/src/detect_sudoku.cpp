//
// Created by jannik on 25.08.2021.
//

#include "detect_sudoku.h"

#include "drawutil.h"
#include "geometry.h"
#include "mathutil.h"

#include <array>
#include <opencv2/imgproc.hpp>
#include <ximgproc_compat.h>

cv::Mat
threshMorphed(const cv::Mat& image, int morphSize)
{
    cv::Mat closingKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morphSize, morphSize));
    cv::Mat closing;
    cv::morphologyEx(image, closing, cv::MORPH_CLOSE, closingKernel);
    cv::Mat noBackgroundImg = (image / closing * 255);

    cv::Mat binImg;
    cv::threshold(noBackgroundImg, binImg, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    return binImg;
}

Contour
locateSudokuContour(const std::vector<Contour>& contours, const cv::Point2f& imageCenter)
{

    double maxArea = 0.0;
    int maxIndex = -1;
    for (auto i = 0; i < contours.size(); i++) {
        const auto& contour = contours.at(i);

        // Is image center contained in polygon?
        auto polytest = cv::pointPolygonTest(contour, imageCenter, false);
        if (polytest < 0)
            continue;

        // Is polygon approximately square?
        std::array<cv::Point2f, 4> points;
        cv::minAreaRect(contour).points(points.data());

        auto d1 = cv::norm(points.at(0) - points.at(1));
        auto d2 = cv::norm(points.at(0) - points.at(2));

        auto squareThresh = (d1 + d2) / 2.0 * 0.5;
        if (abs(d1 - d2) > squareThresh)
            continue;

        // Has polygon got the largest area seen until now?
        auto contourArea = cv::contourArea(contour, false);
        if (contourArea > maxArea) {
            maxArea = contourArea;
            maxIndex = i;
        }
    }
    Contour outputPoints;
    if (maxIndex >= 0) {
        approximateQuad(contours.at(maxIndex), outputPoints, true);
    }

    return outputPoints;
}

bool
detectSudoku(const cv::Mat& sudokuImage, std::array<cv::Point2f, 4>& sudokuCorners, int scaled_side_len)
{
    cv::Mat graySudoku;
    if (sudokuImage.type() != CV_8UC1) {
        // Work in grayscale. Color is not necessary here.
        cv::cvtColor(sudokuImage, graySudoku, cv::COLOR_BGR2GRAY);
    } else {
        graySudoku = sudokuImage;
    }

    // Scale longer side to scaled_side_len
    auto [scale, normSudoku] = resizeMaxSideLen(graySudoku, scaled_side_len);

    // More or less adaptive thresholding but with less noisy results.
    auto sudokuBin = threshMorphed(normSudoku);

    // Slight dilation to fill in holes in the lines
    cv::Mat dilationKernel;
    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat dilated;
    cv::dilate(sudokuBin, dilated, dilationKernel);

    // Contour detection
    std::vector<Contour> contours;
    cv::findContours(dilated, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    cv::Point2f imageCenter(static_cast<float>(normSudoku.cols) / 2.f, static_cast<float>(normSudoku.rows) / 2.f);
    auto sudokuLocation = locateSudokuContour(contours, imageCenter);
    if (sudokuLocation.empty()) {
        // could not locate the sudoku
        return false;
    }

    // Undo scaling performed earlier to obtain sudoku corners in original image space
    sudokuCorners[0] = static_cast<cv::Point2f>(sudokuLocation.at(0)) / scale;
    sudokuCorners[1] = static_cast<cv::Point2f>(sudokuLocation.at(1)) / scale;
    sudokuCorners[2] = static_cast<cv::Point2f>(sudokuLocation.at(2)) / scale;
    sudokuCorners[3] = static_cast<cv::Point2f>(sudokuLocation.at(3)) / scale;

    return true;
}

std::vector<Contour>
findSquares(const cv::Mat& image)
{
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, { 5, 5 }, 0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 5, 5 });

    std::vector<Contour> squares;
    for (int thresh = 0; thresh < 40; thresh += 2) {
        cv::Mat bin;
        if (thresh == 0) {
            cv::Canny(blurred, bin, 0, 50, 5);
            cv::dilate(bin, bin, cv::Mat());
        } else {
            cv::adaptiveThreshold(blurred, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 31, thresh);
            cv::dilate(bin, bin, kernel);
            cv::medianBlur(bin, bin, 3);
        }

        std::vector<Contour> contours;
        cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            double contourLen = cv::arcLength(contour, true);
            Contour approxSquare;
            cv::approxPolyDP(contour, approxSquare, 0.02 * contourLen, true);

            if (approxSquare.size() == 4 && cv::contourArea(approxSquare) > 1000 && cv::isContourConvex(approxSquare)) {

                double maxCos =      //
                  std::max(std::max( //
                             std::abs(angleCos(approxSquare[0], approxSquare[1], approxSquare[2])),
                             std::abs(angleCos(approxSquare[1], approxSquare[2], approxSquare[3]))),
                           std::max( //
                             std::abs(angleCos(approxSquare[2], approxSquare[3], approxSquare[0])),
                             std::abs(angleCos(approxSquare[3], approxSquare[0], approxSquare[1]))));
                if (maxCos < 0.2) {
                    squares.push_back(approxSquare);
                }
            }
        }
    }
    return squares;
}

cv::Mat
binarizeSudoku(const cv::Mat& grayImage)
{
    cv::Mat blurred;
    cv::GaussianBlur(grayImage, blurred, { 5, 5 }, 0);

    cv::Mat sudoku;
    savoulaThreshInv(blurred, sudoku, 255, 51, 0.2);

    const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 5, 5 });
    cv::dilate(sudoku, sudoku, kernel);
    cv::medianBlur(sudoku, sudoku, 7);

    return sudoku;
}

void
detectCellsFast(size_t height, size_t width, std::array<Contour, 81>& cellCoords)
{
    std::array<cv::Point2f, 4> sudokuCorners{
        cv::Point2f(0, 0),
        cv::Point2f(0, height),
        cv::Point2f(width, height),
        cv::Point2f(width, 0),
    };

    std::array<std::array<cv::Point, 10>, 10> gridCorners;
    for (int row = 0; row < 10; ++row) {
        for (int col = 0; col < 10; ++col) {
            float dx = static_cast<float>(row) / static_cast<float>(9);
            cv::Point2f upper = (1.f - dx) * sudokuCorners[0] + dx * sudokuCorners[1];
            cv::Point2f lower = (1.f - dx) * sudokuCorners[3] + dx * sudokuCorners[2];

            float dy = static_cast<float>(col) / static_cast<float>(9);
            cv::Point2f corner = (1.f - dy) * upper + dy * lower;
            gridCorners[row][col] = cv::Point(corner);
        }
    }
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            cellCoords[row * 9 + col].push_back(gridCorners[row][col]);
            cellCoords[row * 9 + col].push_back(gridCorners[row][col + 1]);
            cellCoords[row * 9 + col].push_back(gridCorners[row + 1][col + 1]);
            cellCoords[row * 9 + col].push_back(gridCorners[row + 1][col]);
        }
    }
}

void
detectCells(const cv::Mat& grayImage, std::array<Contour, 81>& cellCoords)
{
    auto sudoku = binarizeSudoku(grayImage);

    std::vector<Contour> contours;
    cv::findContours(sudoku, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    // Select all candidates with area in the given size range and compute the approximate quad.
    for (const auto& contour : contours) {
        Contour cellCandidate;
        approximateQuad(contour, cellCandidate);

        auto area = cv::contourArea(cellCandidate);

        // The box is outside the required size range.
        if (80 * 80 > area || area > 120 * 120) {
            continue;
        }

        // Grid cell in the sudoku occupied by the current cell candidate.
        auto gridPoint = contourCenter(cellCandidate) / (1024 / 9);
        assert(0 <= gridPoint.x && gridPoint.x <= 1024 && 0 <= gridPoint.y && gridPoint.y <= 1024);

        auto nodeIndex = gridPoint.x + 9 * gridPoint.y;
        if (cellCoords.at(nodeIndex).empty()) {
            cellCoords.at(nodeIndex) = cellCandidate;
        } else {
            std::cout << "Cell at (" << gridPoint.x << "," << gridPoint.y << ") already occupied." << std::endl;
        }
    }
}

void
detectCellsRobust(const cv::Mat& grayImage, std::array<Contour, 81>& cellCoords)
{
    std::vector<Contour> squares = findSquares(grayImage);
    std::vector<Contour> goodSquares = convexContourAreaNms(squares, 0.7);

    // Select all candidates with area in the given size range and compute the approximate quad.
    for (auto cellCandidate : goodSquares) {
        normalizeQuadOrientation(cellCandidate, cellCandidate);
        auto area = cv::contourArea(cellCandidate);

        // The box is outside the required size range.
        if (80 * 80 > area || area > 120 * 120) {
            continue;
        }

        // Grid cell in the sudoku occupied by the current cell candidate.
        auto [x, y] = contourCenter(cellCandidate) / (1024 / 9);
        assert(0 <= x && x <= 1024 && 0 <= y && y <= 1024);

        auto nodeIndex = x + 9 * y;
        if (cellCoords.at(nodeIndex).empty()) {
            cellCoords.at(nodeIndex) = cellCandidate;
        } else {
            std::cout << "Cell at (" << x << "," << y << ") already occupied." << std::endl;
        }
    }
}

void
fillMissingSquares(std::array<Contour, 81>& cellCoordinates)
{

    using Polynom = std::vector<float>;
    auto findPoly = [](const std::vector<cv::Point>& points) {
        if (points.size() < 5) { // Require a minimum number of squares per row/col to obtain a robust result.
            return Polynom();
        }

        int polyDegree = 5;
        int numRows = static_cast<int>(points.size()) + 2; // +2 to incorporate the boundary constraint poly' = 0

        cv::Mat a(numRows, polyDegree + 1, CV_32F);
        cv::Mat b(numRows, 1, CV_32F);

        // Build the Vandermonde-matrix-like part of a and the corresponding values of b
        for (int row = 0; row < points.size(); row++) {

            b.at<float>(row, 0) = static_cast<float>(points[row].y);

            float coeff = 1;
            for (int d = 0; d <= polyDegree; d++) {
                a.at<float>(row, d) = coeff;
                coeff = coeff * static_cast<float>(points[row].x);
            }
        }

        // Add the start and end constraints for the polynomial (derivative should be 0.0 at the borders)
        for (int row = 0; row < points.size(); row++) {
            b.at<float>(row, 0) = 0.f;

            a.at<float>(row, 0) = 0.f;
            float coeff = 1;
            for (int d = 1; d <= polyDegree; d++) {
                a.at<float>(row, d) = static_cast<float>(d) * coeff;
                coeff = coeff * static_cast<float>(points[row].x);
            }
        }

        Polynom x;
        cv::solve(a, b, x, cv::DECOMP_QR);

        return x;
    };

    std::array<std::pair<size_t, size_t>, 4> indices{ {
      { 0, 1 }, // upper
      { 3, 2 }, // lower
      { 0, 3 }, // left
      { 1, 2 }  // right
    } };

    std::vector<Polynom> polynoms;
    polynoms.reserve(2 * 9 * 2); // horiz/vert, 9 squares (upper/lower) and (left/right)

    std::vector<cv::Point> points;
    points.reserve(9 * 2); // Max 9 squares per row/col with two points each per point row/col

    for (auto [i1, i2] : indices) {
        for (size_t row = 0; row < 9; row++) {
            points.clear();
            for (size_t col = 0; col < 9; col++) {
                const Contour& c = cellCoordinates[9 * row + col];
                if (!c.empty()) {
                    points.push_back(c[i1]);
                    points.push_back(c[i2]);
                }
            }

            polynoms.push_back(findPoly(points));
        }
    }
}