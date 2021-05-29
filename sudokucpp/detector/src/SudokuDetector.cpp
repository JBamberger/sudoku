#include <SudokuDetector.h>

#include <SudokuSolver.h>
#include <drawutil.h>
#include <mathutil.h>
#include <utils.h>

#include <algorithm>
#include <array>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/ximgproc.hpp>
#include <ximgproc_compat.h>

struct SudokuDetector::Impl
{
    const int scaled_side_len = 1024;
    CellClassifier cellClassifier;

    explicit Impl(const std::string& classifierPath): cellClassifier(classifierPath) {}

    std::unique_ptr<SudokuDetection> detect(const cv::Mat& sudokuImage)
    {
        // Data class to hold the results
        auto detection = std::make_unique<SudokuDetection>();

        if (!detectSudoku(sudokuImage, detection->sudokuCorners)) {
            return detection;
        }

        const cv::Size scaledSudokuSize(scaled_side_len, scaled_side_len);
        detection->unwarpTransform = getUnwarpTransform(detection->sudokuCorners, scaledSudokuSize);
        detection->foundSudoku = true;

        cv::Mat warped;
        cv::warpPerspective(sudokuImage, warped, detection->unwarpTransform, scaledSudokuSize, cv::INTER_AREA);

        detectCells(warped, detection->cellCoords);
        classifyCells(warped, detection->cellCoords, detection->cellLabels);
        detection->foundAllCells =
          std::all_of(std::begin(detection->cellLabels), std::end(detection->cellLabels), [](int i) { return i >= 0; });

        if (detection->foundAllCells) {
            SudokuSolver solver;

            detection->solution = solver.solve(detection->cellLabels);
        }

        //        cv::Mat canvas = warped.clone();
        ////        for (int i = 0; i < 81; i++) {
        ////            const auto& contour = detection->cellCoords.at(i);
        ////            if (contour.empty()) {
        ////                continue;
        ////            }
        ////
        ////            drawOrientedRect(canvas, contour);
        ////
        ////            const auto msg = std::to_string(detection->cellLabels.at(i));
        ////            const auto center = contourCenter(contour);
        ////            drawCenteredText(canvas, msg, center);
        ////        }
        //
        //        cv::Mat overlay = detection->renderOverlay(scaled_side_len, scaled_side_len);
        //        cv::Mat composite = compositeImage(canvas, overlay);
        //
        //        cv::imshow("Contours", composite);
        //        cv::waitKey();

        return detection;
    }

    bool detectSudoku(const cv::Mat& sudokuImage, std::array<cv::Point2f, 4>& sudokuCorners) const
    {
        // Work in grayscale. Color is not necessary here.
        cv::Mat graySudoku;
        cv::cvtColor(sudokuImage, graySudoku, cv::COLOR_BGR2GRAY);

        // Scale longer side to scaled_side_len
        auto [scale, normSudoku] = resizeMaxSideLen(graySudoku, scaled_side_len);

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

        std::vector<cv::Point> outputPoints;
        if (maxIndex >= 0) {
            approximateQuad(contours.at(maxIndex), outputPoints, true);
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

    void classifyCells(const cv::Mat& image,
                       std::array<std::vector<cv::Point>, 81>& cellCoordinates,
                       std::array<int, 81>& cellLabels)
    {
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

        int pad = 3;
        int patchSize = 64;
        cv::Size paddedSize(patchSize + 2 * pad, patchSize + 2 * pad);
        const cv::Rect2i cropRect(pad, pad, patchSize, patchSize);

        std::fill(std::begin(cellLabels), std::end(cellLabels), -1);
        for (int i = 0; i < 81; i++) {
            const auto& coordinates = cellCoordinates.at(i);

            if (!coordinates.empty()) {
                std::array<cv::Point2f, 4> transformTarget{
                    static_cast<cv::Point2f>(coordinates.at(0)),
                    static_cast<cv::Point2f>(coordinates.at(1)),
                    static_cast<cv::Point2f>(coordinates.at(2)),
                    static_cast<cv::Point2f>(coordinates.at(3)),
                };

                const auto paddedCellPatch = unwarpPatch(grayImage, transformTarget, paddedSize);
                const auto cellPatch = paddedCellPatch(cropRect);
                cellLabels[i] = cellClassifier.classify(cellPatch);
            }
        }
    }

    static void detectCells(const cv::Mat& image, std::array<std::vector<cv::Point>, 81>& cellCoords)
    {
        auto sudoku = binarizeSudoku(image);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(sudoku, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        // Select all candidates with area in the given size range and compute the approximate quad.
        for (const auto& contour : contours) {
            std::vector<cv::Point> cellCandidate;
            approximateQuad(contour, cellCandidate);

            auto area = cv::contourArea(cellCandidate);

            // The box is outside of the required size range.
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

    [[nodiscard]] static cv::Mat binarizeSudoku(const cv::Mat& image)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, { 5, 5 }, 0);

        cv::Mat sudoku;
        //        cv::ximgproc::niBlackThreshold(
        //          blurred, sudoku, 255, cv::THRESH_BINARY_INV, 51, 0.2, cv::ximgproc::BINARIZATION_SAUVOLA);
        savoulaThreshInv(blurred, sudoku, 255, 51, 0.2);

        const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 5, 5 });
        cv::dilate(sudoku, sudoku, kernel);
        cv::medianBlur(sudoku, sudoku, 7);

        return sudoku;
    }
};

SudokuDetector::SudokuDetector(const std::string& classifierPath)
  : pimpl(std::make_unique<Impl>(classifierPath))
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
