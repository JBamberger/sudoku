#include <SudokuDetector.h>

#include "detect_sudoku.h"
#include <SudokuSolver.h>
#include <algorithm>
#include <array>
#include <drawutil.h>
#include <mathutil.h>

[[nodiscard]] cv::Mat
getUnwarpTransform(const std::array<cv::Point2f, 4>& corners, const cv::Size& outSize)
{
    auto w = static_cast<float>(outSize.width);
    auto h = static_cast<float>(outSize.height);
    std::array<cv::Point2f, 4> destinations{
        cv::Point2f(0.f, 0.f),
        cv::Point2f(w, 0.f),
        cv::Point2f(w, h),
        cv::Point2f(0.f, h),
    };
    return cv::getPerspectiveTransform(corners, destinations);
}

struct SudokuDetector::Impl
{
    const int scaled_side_len = 1024;
    std::unique_ptr<CellClassifier> cellClassifier;
    std::unique_ptr<SudokuSolver> sudokuSolver;

    explicit Impl(const std::string& classifierPath)
      : cellClassifier(std::make_unique<CellClassifier>(classifierPath))
      , sudokuSolver(SudokuSolver::create(SolverType::Dlx))
    {
    }

    [[nodiscard]] std::unique_ptr<SudokuDetection> detect(const cv::Mat& sudokuImage) const
    {
        cv::Mat graySudoku;
        cv::cvtColor(sudokuImage, graySudoku, cv::COLOR_BGR2GRAY);

        // Data class to hold the results
        auto detection = std::make_unique<SudokuDetection>();

        if (!detectSudoku(graySudoku, detection->sudokuCorners, scaled_side_len)) {
            return detection;
        }

        const cv::Size scaledSudokuSize(scaled_side_len, scaled_side_len);
        detection->unwarpTransform = getUnwarpTransform(detection->sudokuCorners, scaledSudokuSize);
        detection->foundSudoku = true;

        cv::Mat warped;
        cv::warpPerspective(graySudoku, warped, detection->unwarpTransform, scaledSudokuSize, cv::INTER_AREA);

        // detectCells(warped, detection->cellCoords);
        detectCellsRobust(warped, detection->cellCoords);
        // detectCellsFast(scaled_side_len, scaled_side_len, detection->cellCoords);

        classifyCells(warped, detection->cellCoords, detection->cellLabels);
        detection->foundAllCells =
          std::all_of(std::begin(detection->cellLabels), std::end(detection->cellLabels), [](int i) { return i >= 0; });

        if (detection->foundAllCells) {
            SudokuGrid grid(9);
            for (int i = 0; i < 81; i++) {
                grid.at(i) = detection->cellLabels[i];
            }
            auto solvedGrid = sudokuSolver->solve(grid);
            if (solvedGrid) {
                detection->solution = std::make_unique<std::array<int, 81>>();
                for (int i = 0; i < 81; i++) {
                    detection->solution->at(i) = solvedGrid->at(i);
                }
            } else {
                detection->solution = nullptr;
            }
        }

        return detection;
    }

    void classifyCells(const cv::Mat& grayImage,
                       const std::array<std::vector<cv::Point>, 81>& cellCoordinates,
                       std::array<int, 81>& cellLabels) const
    {
        const int pad = 3;
        const int patchSize = 64;
        const cv::Size paddedSize(patchSize + 2 * pad, patchSize + 2 * pad);
        const cv::Rect2i cropRect(pad, pad, patchSize, patchSize);

        const size_t numCells = cellCoordinates.size();
        std::vector<size_t> indices;
        indices.reserve(numCells);
        std::vector<cv::Mat> patches;
        patches.reserve(numCells);
        for (size_t i = 0; i < numCells; i++) {
            const auto& coordinates = cellCoordinates.at(i);

            if (coordinates.empty())
                continue;

            std::array<cv::Point2f, 4> transformTarget{
                static_cast<cv::Point2f>(coordinates.at(0)),
                static_cast<cv::Point2f>(coordinates.at(1)),
                static_cast<cv::Point2f>(coordinates.at(2)),
                static_cast<cv::Point2f>(coordinates.at(3)),
            };
            const cv::Mat m = getUnwarpTransform(transformTarget, paddedSize);

            cv::Mat paddedCellPatch;
            cv::warpPerspective(grayImage, paddedCellPatch, m, paddedSize, cv::INTER_AREA);

            patches.push_back(paddedCellPatch(cropRect).clone());
            indices.push_back(i);
        }

        std::fill(std::begin(cellLabels), std::end(cellLabels), -1);
        if (patches.empty()) {
            return;
        }

        auto labels = cellClassifier->classify(patches);
        for (size_t i = 0; i < indices.size(); i++) {
            cellLabels[indices[i]] = labels[i];
        }
    }
};

SudokuDetector::SudokuDetector(const std::string& classifierPath)
  : pimpl(std::make_unique<Impl>(classifierPath))
{
}

SudokuDetector::SudokuDetector(SudokuDetector&&) noexcept = default;

SudokuDetector::~SudokuDetector() = default;

SudokuDetector&
SudokuDetector::operator=(SudokuDetector&&) noexcept = default;

std::unique_ptr<SudokuDetection>
SudokuDetector::detect(const cv::Mat& sudokuImage)
{
    return pimpl->detect(sudokuImage);
}
