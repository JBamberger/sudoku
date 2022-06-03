#include <drawutil.h>
#include <geometry.h>
#include <utils.h>

#include "cli_opts.h"
#include "detect_sudoku.h"
#include "ximgproc_compat.h"
#include <chrono>
#include <filesystem>
#include <fmt/format.h>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

struct DetectionResult
{
    double time_ms = 0.0;
    bool success = false;
    std::array<cv::Point2f, 4> sudokuCorners;
    double iou = 0.0;
};

auto
test(const SudokuGroundTruth& gtSudoku, DetectionResult& result) -> void
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
loop(const std::vector<SudokuGroundTruth>& groundTruth, std::vector<DetectionResult>& results, double thresh = 0.9)
  -> std::vector<size_t>
{
    {
        auto result = results.begin();
        auto gt = groundTruth.begin();
        for (; gt != groundTruth.end() && result != results.end(); gt++, result++) {
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
    double time = 0;
    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        if (result.success) {
            if (result.iou > thresh) {
                successCount++;
            } else {
                iouFailCount++;
                failures.push_back(i);
            }
            time += result.time_ms;
        } else {
            detFailCount++;
            failures.push_back(i);
        }
    }

    fmt::print("#########################################\n"
               "Successful:       {:d}\n"
               "Detection failed: {:d}\n"
               "IoU failed:       {:d}\n"
               "Mean time:        {:.01f}ms\n"
               "#########################################\n",
               successCount,
               detFailCount,
               iouFailCount,
               time / (successCount + detFailCount));
    return failures;
}

auto
showFailures(const std::vector<SudokuGroundTruth>& groundTruth,
             const std::vector<DetectionResult>& results,
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
runInteractive(const fs::path& fileRoot, const fs::path& gtPath) -> void
{
    auto gt = readGroundTruth(fileRoot, gtPath);

    std::cout << "OpenCV Version: " << cv::getVersionString() << std::endl;
    std::cout << "Found " << gt.size() << " ground truth entries" << std::endl;

    std::vector<DetectionResult> results(gt.size());
    const auto failures = loop(gt, results);

    showFailures(gt, results, failures);
}

auto
runNonInteractive(const fs::path& fileRoot, const fs::path& gtPath) -> void
{
    auto gt = readGroundTruth(fileRoot, gtPath);
    std::vector<DetectionResult> results(gt.size());
    loop(gt, results);
}

auto
runBenchmark(const fs::path& fileRoot, const fs::path& gtPath, size_t sudokuIndex, size_t numIterations) -> void
{
    volatile double a = 0;
    auto gt = readGroundTruth(fileRoot, gtPath);
    const auto sudoku = gt.at(sudokuIndex);
    for (size_t i = 0; i < numIterations; i++) {
        DetectionResult result;
        test(sudoku, result);
        a += result.iou;
    }
}

auto
main(int argc, char* argv[]) -> int
{
    CliOptionsParser parser(argc, argv);

    const std::string& rootStr = parser.getPositionalOption(0);
    if (rootStr.empty()) {
        std::cerr << "Missing path to sources. Call as <benchmark-exe>.exe <path-to-sources>" << std::endl;
        exit(1);
    }
    fs::path root(rootStr);
    fs::path gtPath = root / "data" / "sudokus" / "annotations" / "sudoku_bounds.csv";
    if (!fs::exists(gtPath) || !fs::is_regular_file(gtPath)) {
        std::cerr << "The specified gt path is not a file." << std::endl;
        exit(1);
    }

    const std::string& mode = parser.getOption("--mode");
    if (mode.empty() || mode == "interactive") {
        runInteractive(root, gtPath);
    } else if (mode == "static") {
        runNonInteractive(root, gtPath);
    } else if (mode == "benchmark") {
        size_t sudokuIndex = 0;
        size_t numIterations = 1000;
        runBenchmark(root, gtPath, sudokuIndex, numIterations);
    } else {
        std::cerr << "Invalid mode: '" << mode << "'. Must be one of 'interactive' or 'static'." << std::endl;
        exit(1);
    }
}
