#include <drawutil.h>
#include <geometry.h>
#include <utils.h>

#include <chrono>
#include <detect_sudoku.h>
#include <filesystem>
#include <fmt/format.h>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

struct Result
{
    double time_ms = 0.0;
    bool success = false;
    std::array<cv::Point2f, 4> sudokuCorners;
    double iou = 0.0;
};

auto
test(const SudokuGroundTruth& gtSudoku, Result& result) -> void
{
    using namespace std::chrono;

    const auto sudokuImg = cv::imread(gtSudoku.imagePath.string(), cv::IMREAD_COLOR);

    time_point beginDetect = steady_clock::now();
    result.success = detectSudoku(sudokuImg, result.sudokuCorners, 1024);
    time_point endDetect = steady_clock::now();

    result.time_ms = static_cast<double>(duration_cast<microseconds>(endDetect - beginDetect).count()) / 1000.0;
    if (result.success) {
        result.iou = convexContourIoU(asContour(result.sudokuCorners), asContour(gtSudoku.bounds.corners));
    } else {
        result.iou = 0;
    }
}

auto
loop(const std::vector<SudokuGroundTruth>& groundTruth, std::vector<Result>& results, double thresh = 0.9)
  -> std::vector<size_t>
{
    {
        auto result = results.begin();
        for (auto gt = groundTruth.begin(); gt < groundTruth.end(); gt++, result++) {
            test(*gt, *result);

            fmt::print("success={:d}; iou={:.03f}; time={:5.01f}ms {}\n",
                       result->iou > thresh,
                       result->iou,
                       result->time_ms,
                       gt->imagePath.string());
            std::cout << std::flush;
        }
    }

    std::vector<size_t> failures;

    int successCount = 0, detFailCount = 0, iouFailCount = 0;
    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        if (result.success) {
            if (result.iou > thresh) {
                successCount++;
            } else {
                iouFailCount++;
                failures.push_back(i);
            }
        } else {
            detFailCount++;
            failures.push_back(i);
        }
    }

    fmt::print("#########################################\n"
               "Successful:       {:d}\n"
               "Detection failed: {:d}\n"
               "IoU failed:       {:d}\n"
               "#########################################\n",
               successCount,
               detFailCount,
               iouFailCount);
    return failures;
}

auto
showFailures(const std::vector<SudokuGroundTruth>& groundTruth,
             const std::vector<Result>& results,
             const std::vector<size_t>& failures) -> void
{
    if (failures.empty()) {
        return;
    }

    size_t sudokuNum = 0;
    while (true) {
        const auto& index = failures[sudokuNum];
        const auto& result = results[index];
        const auto& gt = groundTruth[index];

        const auto imgPath = gt.imagePath.string();
        std::cout << fmt::format("Sudoku {:d} Path={:s}\n", sudokuNum, imgPath) << std::flush;
        auto canvas = cv::imread(imgPath, cv::IMREAD_COLOR);

        if (result.success) {
            drawOrientedRect(canvas, asContour(result.sudokuCorners), { 0, 0, 255 });
            drawOrientedRect(canvas, asContour(gt.bounds.corners), { 0, 255, 0 });
        }

        auto [scale, resizedCanvas] = resizeMaxSideLen(canvas, 1024);
        cv::imshow("Sudoku", resizedCanvas);
        int key = cv::waitKey();

        switch (key) {
            case -1:
            case 'q':
                return;
            case 'n':
                sudokuNum++;
                break;
            case 'p':
                sudokuNum--;
                break;
            default:
                sudokuNum++;
        }
        if (sudokuNum >= failures.size()) {
            sudokuNum = failures.size() - 1;
            std::cout << "Reached last Sudoku. Press q to quit." << std::endl;
        } else if (sudokuNum < 0) {
            sudokuNum = 0;
            std::cout << "Reached first Sudoku. Press q to quit." << std::endl;
        }
    }
}

auto
main(int argc, char* argv[]) -> int
{
    if (argc < 2) {
        std::cerr << "Missing path to sources. Call as <benchmark-exe>.exe <path-to-sources>" << std::endl;
        exit(1);
    }

    fs::path root(argv[1]);
    fs::path gtPath = root / "data" / "sudokus" / "annotations" / "sudoku_bounds.csv";

    if (!fs::exists(gtPath) || !fs::is_regular_file(gtPath)) {
        std::cerr << "The specified gt path is not a file." << std::endl;
        exit(1);
    }

    auto gt = readGroundTruth(root, gtPath);

    std::cout << "OpenCV Version: " << cv::getVersionString() << std::endl;
    std::cout << "Found " << gt.size() << " ground truth entries" << std::endl;

    std::vector<Result> results(gt.size());
    const auto failures = loop(gt, results);

    showFailures(gt, results, failures);
}
