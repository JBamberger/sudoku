#include <SudokuDetector.h>

#include <mathutil.h>
#include <utils.h>

#include <algorithm>
#include <array>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

struct SudokuDetector::Impl
{
    const int scaled_side_len = 1024;
    CellClassifier cellClassifier;

    std::unique_ptr<SudokuDetection> detect(const cv::Mat& sudokuImage)
    {
        // Data class to hold the results
        auto detection = std::make_unique<SudokuDetection>();

        if (!detectSudoku(sudokuImage, detection->sudokuCorners)) {
            return detection;
        }

        const cv::Size scaledSudokuSize(scaled_side_len, scaled_side_len);
        const auto unwarpTransform = getUnwarpTransform(detection->sudokuCorners, scaledSudokuSize);

        cv::Mat warped;
        cv::warpPerspective(sudokuImage, warped, unwarpTransform, scaledSudokuSize, cv::INTER_AREA);

        detection->unwarpTransform = unwarpTransform;
        detection->foundSudoku = true;

        auto [cellPatches, cellCoords] = extractCells(warped);

        std::array<int, 81> sudokuGrid{};
        std::fill(std::begin(sudokuGrid), std::end(sudokuGrid), 0);
        for (int i = 0; i < 81; i++) {
            const auto& cellPatch = cellPatches.at(i);

            if (cellPatch.data == nullptr) {
                std::cerr << "Could not locate cell patch: " << i << std::endl;
                sudokuGrid[i] = 0;
            } else {
                cv::Mat grayCellPatch;
                cv::cvtColor(cellPatch, grayCellPatch, cv::COLOR_BGR2GRAY);
                sudokuGrid[i] = cellClassifier.classify(grayCellPatch);
            }
        }

        cv::Mat canvas = warped.clone();
        for (int i = 0; i < 81; i++) {
            const auto& contour = cellCoords.at(i);
            if (contour.empty()) {
                continue;
            }
            cv::line(canvas, contour.at(0), contour.at(1), { 255, 0, 0 }, 2);
            cv::line(canvas, contour.at(1), contour.at(2), { 0, 0, 255 }, 2);
            cv::line(canvas, contour.at(2), contour.at(3), { 0, 255, 0 });
            cv::line(canvas, contour.at(3), contour.at(0), { 0, 255, 0 });

            const auto p = (contour.at(0) + contour.at(1) + contour.at(2) + contour.at(3)) / 4;

            const auto msg = std::to_string(sudokuGrid.at(i));
            const auto fontFace = cv::FONT_HERSHEY_SIMPLEX;
            const auto fontScale = 2.0;
            const auto fontThickness = 3;

            int baseline = 0;
            auto textSize = cv::getTextSize(msg, fontFace, fontScale, fontThickness, &baseline);
            baseline += fontThickness;

            cv::Point msgOrigin(p.x - textSize.width / 2, p.y + textSize.height / 2);
            cv::putText(canvas, msg, msgOrigin, fontFace, fontScale, { 0, 255, 0 }, fontThickness);
        }
        cv::imshow("Contours", canvas);
        cv::waitKey();

        return detection;
    }

    [[nodiscard]] std::tuple<cv::Mat, double> inputResize(const cv::Mat& sudokuImage) const
    {
        int h = sudokuImage.rows;
        int w = sudokuImage.cols;

        double scale = static_cast<double>(scaled_side_len) / static_cast<double>(h > w ? h : w);

        cv::Mat scaledImg;
        cv::resize(sudokuImage, scaledImg, cv::Size(), scale, scale, cv::INTER_AREA);

        return std::make_tuple(scaledImg, scale);
    }

    bool detectSudoku(const cv::Mat& sudokuImage, std::array<cv::Point2f, 4>& sudokuCorners) const
    {
        // Work in grayscale. Color is not necessary here.
        cv::Mat graySudoku;
        cv::cvtColor(sudokuImage, graySudoku, cv::COLOR_BGR2GRAY);

        // Scale longer side to scaled_side_len
        int h = sudokuImage.rows;
        int w = sudokuImage.cols;
        double scale = static_cast<double>(scaled_side_len) / static_cast<double>(h > w ? h : w);
        cv::Mat normSudoku;
        cv::resize(graySudoku, normSudoku, cv::Size(), scale, scale, cv::INTER_AREA);

        // More or less adaptive thresholding but with less noisy results.
        cv::Mat closingKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25));
        cv::Mat closing;
        cv::morphologyEx(normSudoku, closing, cv::MORPH_CLOSE, closingKernel);
        normSudoku = (normSudoku / closing * 255);

        cv::Mat sudokuBin;
        cv::threshold(normSudoku, sudokuBin, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

        // Slight dilation to fill in holes in the lines
        cv::Mat dilationKernel;
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::Mat dilated;
        cv::dilate(sudokuBin, dilated, dilationKernel);

        // Contour detection
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        cv::Point2f imageCenter(static_cast<float>(normSudoku.cols) / 2.f, static_cast<float>(normSudoku.rows) / 2.f);
        auto sudokuLocation = locateSudokuContour(contours, imageCenter);
        if (sudokuLocation.empty()) {
            // could not localize the sudoku
            return false;
        }

        // Undo scaling performed earlier to obtain sudoku corners in original image space
        sudokuCorners[0] = static_cast<cv::Point2f>(sudokuLocation.at(0)) / scale;
        sudokuCorners[1] = static_cast<cv::Point2f>(sudokuLocation.at(1)) / scale;
        sudokuCorners[2] = static_cast<cv::Point2f>(sudokuLocation.at(2)) / scale;
        sudokuCorners[3] = static_cast<cv::Point2f>(sudokuLocation.at(3)) / scale;

        return true;
    }

    static std::vector<cv::Point> locateSudokuContour(const std::vector<std::vector<cv::Point>>& contours,
                                                      const cv::Point2f& imageCenter)
    {

        double maxArea = 0.0;
        int maxIndex = -1;
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

    static void approximateQuad(const std::vector<cv::Point>& contour,
                                std::vector<cv::Point>& outRect,
                                bool normalizeOrientation = true)
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

    static void normalizeQuadOrientation(const std::vector<cv::Point>& contour, std::vector<cv::Point>& outRect)
    {
        auto sortX = argsort<cv::Point>(contour, [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
        auto sortY = argsort<cv::Point>(contour, [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });

        std::array<size_t, 4> bins{ 0, 0, 0, 0 };
        bins[sortX[0]] += 1;
        bins[sortX[1]] += 1;
        bins[sortY[0]] += 1;
        bins[sortY[1]] += 1;
        //    std::cout << bins[0] << ", " << bins[1] << ", " << bins[2] << ", " << bins[3] << std::endl;

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
            auto index = i + offset;
            if (index >= contour.size()) {
                index -= static_cast<int>(contour.size());
            }
            rotatedBox.push_back(contour.at(index));
        }

        auto delta = rotatedBox.at(1) - rotatedBox.at(0);
        auto angle = orientedAngle<double>(static_cast<double>(delta.x), static_cast<double>(delta.y), 1.0, 0.0);
        angle = angle / M_PI * 180.0;
        //    std::cout << angle << std::endl;

        outRect.at(0) = rotatedBox.at(0);
        if (angle < -45.0 || 45.0 < angle) {
            for (int i = 1; i < rotatedBox.size(); i++) {
                outRect.at(i) = rotatedBox.at(rotatedBox.size() - i);
            }
        } else {
            for (int i = 1; i < rotatedBox.size(); i++) {
                outRect.at(i) = rotatedBox.at(i);
            }
        }
    }

    [[nodiscard]] static cv::Mat unwarpPatch(const cv::Mat& image,
                                             const std::array<cv::Point2f, 4>& corners,
                                             const cv::Size& outSize)
    {
        cv::Mat M = getUnwarpTransform(corners, outSize);

        cv::Mat output;
        cv::warpPerspective(image, output, M, outSize, cv::INTER_AREA);

        return output;
    }
    [[nodiscard]] static cv::Mat getUnwarpTransform(const std::array<cv::Point2f, 4>& corners, const cv::Size& outSize)
    {
        auto w = static_cast<float>(outSize.width);
        auto h = static_cast<float>(outSize.height);
        std::array<cv::Point2f, 4> destinations{
            cv::Point2f(0.f, 0.f),
            cv::Point2f(w, 0.f),
            cv::Point2f(w, h),
            cv::Point2f(0.f, h),
        };
        auto M = cv::getPerspectiveTransform(corners, destinations);
        return M;
    }

    static std::pair<std::vector<cv::Mat>, std::vector<std::vector<cv::Point>>> extractCells(const cv::Mat& image)
    {
        auto sudoku = binarizeSudoku(image);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(sudoku, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::pair<cv::Point, std::vector<cv::Point>>> cells;
        cells.reserve(contours.size());
        for (const auto& contour : contours) {
            std::vector<cv::Point> box;
            approximateQuad(contour, box);

            auto area = cv::contourArea(box);
            if (80 * 80 <= area && area <= 120 * 120) {
                cv::Point center(0, 0);
                for (const auto& point : box) {
                    center.x += point.x;
                    center.y += point.y;
                }
                center /= static_cast<int>(box.size());
                cells.emplace_back(center, box);
            }
        }

        int step = 1024 / 9;
        std::array<int, 81> cellToNode{};
        cellToNode.fill(-1);
        for (int cellIndex = 0; cellIndex < cells.size(); ++cellIndex) {
            const auto& pair = cells.at(cellIndex);
            auto center = pair.first;
            auto cell = pair.second;

            auto gridPoint = center / step;
            if (!(0 <= gridPoint.x && gridPoint.x <= 1024 && 0 <= gridPoint.y && gridPoint.y <= 1024)) {
                exit(-1);
            }

            auto p = gridPoint.x + 9 * gridPoint.y;
            if (cellToNode.at(p) < 0) {
                cellToNode.at(p) = cellIndex;
            } else {
                std::cout << "Cell at (" << gridPoint.x << "," << gridPoint.y << ") already occupied by "
                          << cellToNode.at(p) << '.' << std::endl;
            }
        }

        int pad = 3;
        int patchSize = 64;
        cv::Size paddedSize(patchSize + 2 * pad, patchSize + 2 * pad);
        const cv::Rect2i cropRect(pad, pad, patchSize, patchSize);

        std::vector<std::vector<cv::Point>> cellCoordinates;
        cellCoordinates.reserve(81);
        std::vector<cv::Mat> cellPatches;
        for (int i = 0; i < 81; i++) {
            int nodeIndex = cellToNode.at(i);

            std::vector<cv::Point> coordinates;
            cv::Mat cellPatch;

            if (nodeIndex >= 0) {
                coordinates = cells.at(nodeIndex).second;

                std::array<cv::Point2f, 4> transformTarget{
                    static_cast<cv::Point2f>(coordinates.at(0)),
                    static_cast<cv::Point2f>(coordinates.at(1)),
                    static_cast<cv::Point2f>(coordinates.at(2)),
                    static_cast<cv::Point2f>(coordinates.at(3)),
                };

                cv::Mat paddedCellPatch = unwarpPatch(image, transformTarget, paddedSize);
                cellPatch = paddedCellPatch(cropRect);
            }
            cellCoordinates.push_back(coordinates);
            cellPatches.push_back(cellPatch);
        }
        return std::make_pair(cellPatches, cellCoordinates);
    }

    [[nodiscard]] static cv::Mat binarizeSudoku(const cv::Mat& image)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, { 5, 5 }, 0);

        cv::Mat sudoku;
        cv::ximgproc::niBlackThreshold(
          blurred, sudoku, 255, cv::THRESH_BINARY_INV, 51, 0.2, cv::ximgproc::BINARIZATION_SAUVOLA);

        const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 5, 5 });
        cv::dilate(sudoku, sudoku, kernel);
        cv::medianBlur(sudoku, sudoku, 7);

        return sudoku;
    }
};

SudokuDetector::SudokuDetector()
  : pimpl(std::make_unique<Impl>())
{}

SudokuDetector::SudokuDetector(SudokuDetector&&) noexcept = default;

SudokuDetector::~SudokuDetector() = default;

SudokuDetector&
SudokuDetector::operator=(SudokuDetector&&) noexcept = default;

std::unique_ptr<SudokuDetection>
SudokuDetector::detect(const cv::Mat& sudokuImage)
{
    return pimpl->detect(sudokuImage);
}
