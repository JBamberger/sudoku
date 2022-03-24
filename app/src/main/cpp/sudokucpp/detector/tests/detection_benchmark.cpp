#include <drawutil.h>
#include <utils.h>

#include <detect_sudoku.h>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

std::vector<SudokuGroundTruth>
loop(const std::vector<SudokuGroundTruth>& groundTruth, double thresh = 0.9)
{
    int successCount = 0;
    int detFailCount = 0;
    int iouFailCount = 0;

    std::vector<SudokuGroundTruth> failures;

    for (const auto& gtSudoku : groundTruth) {
        cv::Mat sudokuImg = cv::imread(gtSudoku.imagePath.string(), cv::IMREAD_COLOR);

        std::array<cv::Point2f, 4> sudokuCorners;
        if (detectSudoku(sudokuImg, sudokuCorners, 1024)) {
            double iou = contourIoU(asContour(sudokuCorners), asContour(gtSudoku.bounds.corners));
            std::cout << (iou > thresh) << ";  " << iou << " ";
            if (iou > thresh) {
                successCount++;
            } else {
                iouFailCount++;
                failures.push_back(gtSudoku);
            }
        } else {
            std::cout << "Failed to detect! ";
            detFailCount++;
            failures.push_back(gtSudoku);
        }
        std::cout << gtSudoku.imagePath << std::endl;
    }

    std::cout << "#########################################" << std::endl;
    std::cout << "Successful:       " << successCount << std::endl;
    std::cout << "Detection failed: " << detFailCount << std::endl;
    std::cout << "IoU failed:       " << iouFailCount << std::endl;
    std::cout << "#########################################" << std::endl;

    return failures;
}

void
showFailures(const std::vector<SudokuGroundTruth>& failures)
{
    if (failures.empty()) {
        return;
    }

    size_t sudokuNum = 0;
    while (true) {
        const auto& gtSudoku = failures[sudokuNum];
        std::cout << "Sudoku " << sudokuNum << " Path: " << gtSudoku.imagePath.string() << std::endl;

        cv::Mat sudokuImg = cv::imread(gtSudoku.imagePath.string(), cv::IMREAD_COLOR);

        cv::Mat canvas = sudokuImg.clone();

        std::array<cv::Point2f, 4> sudokuCorners;
        if (detectSudoku(sudokuImg, sudokuCorners, 1024)) {
            drawOrientedRect(canvas, asContour(sudokuCorners));
        }

        //        auto detection = detectSudoku(sudokuImg);
        //        if (detection->foundSudoku) {
        //            detection->drawOverlay(canvas);
        //        } else{
        //            std::cerr << "Failed to detect sudoku!" << std::endl;
        //        }

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
                // std::cout << "Pressed key: " << key << std::endl;
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

int
main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Missing path to sources. Call as <benchmark-exe>.exe <path-to-sources>" << std::endl;
        exit(1);
    }

    fs::path root(argv[1]);
    fs::path gtPath = root / "data" / "sudokus" / "ground_truth_new.csv";

    if (!fs::exists(gtPath) || !fs::is_regular_file(gtPath)) {
        std::cerr << "The specified gt path is not a file." << std::endl;
        exit(1);
    }

    auto gt = readGroundTruth(root, gtPath);

    std::cout << "OpenCV Version: " << cv::getVersionString() << std::endl;
    std::cout << "Found " << gt.size() << " ground truth entries" << std::endl;

    const auto failures = loop(gt);

    showFailures(failures);
}
