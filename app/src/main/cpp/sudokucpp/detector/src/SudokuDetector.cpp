#include <SudokuDetector.h>

#include <SudokuSolver.h>
#include <detect_sudoku.h>
#include <drawutil.h>
#include <mathutil.h>

#include <algorithm>
#include <array>

struct SudokuDetector::Impl
{
    const int scaled_side_len = 1024;
    CellClassifier cellClassifier;

    explicit Impl(const std::string& classifierPath)
      : cellClassifier(classifierPath)
    {}

    std::unique_ptr<SudokuDetection> detect(const cv::Mat& sudokuImage)
    {
        // Data class to hold the results
        auto detection = std::make_unique<SudokuDetection>();

        if (!detectSudoku(sudokuImage, detection->sudokuCorners, scaled_side_len)) {
            return detection;
        }

        const cv::Size scaledSudokuSize(scaled_side_len, scaled_side_len);
        detection->unwarpTransform = getUnwarpTransform(detection->sudokuCorners, scaledSudokuSize);
        detection->foundSudoku = true;

        cv::Mat warped;
        cv::warpPerspective(sudokuImage, warped, detection->unwarpTransform, scaledSudokuSize, cv::INTER_AREA);

        // detectCells(warped, detection->cellCoords);
        detectCellsRobust(warped, detection->cellCoords);
        // detectCellsFast(scaled_side_len, scaled_side_len, detection->cellCoords);

        classifyCells(warped, detection->cellCoords, detection->cellLabels);
        detection->foundAllCells =
          std::all_of(std::begin(detection->cellLabels), std::end(detection->cellLabels), [](int i) { return i >= 0; });

        if (detection->foundAllCells) {
            auto solver = SudokuSolver::create(SolverType::Dlx);

            SudokuGrid grid(9);
            for (int i = 0; i < 81; i++) {
                grid.at(i) = detection->cellLabels[i];
            }
            auto solvedGrid = solver->solve(grid);
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

        std::vector<size_t> indices;
        indices.reserve(81);
        std::vector<cv::Mat> patches;
        patches.reserve(81);
        for (size_t i = 0; i < 81; i++) {
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

                patches.push_back(cellPatch);
                indices.push_back(i);
            }
        }

        std::fill(std::begin(cellLabels), std::end(cellLabels), -1);

        if (!patches.empty()) {
            auto labels = cellClassifier.classify(patches);
            for (size_t i = 0; i < indices.size(); i++) {
                cellLabels[indices[i]] = labels[i];
            }
        }
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
