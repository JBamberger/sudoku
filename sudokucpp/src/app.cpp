#include <SudokuDetector.h>

#include <config.h>
#include <utils.h>

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>

namespace fs = std::filesystem;

void
processSudoku(const fs::path& path, const Quad& gt_bbox)
{
    cv::Mat sudokuImg = cv::imread(path.string(), cv::IMREAD_COLOR);
    auto detector = SudokuDetector();
    auto detection = detector.detect(sudokuImg);
}

int
main()
{
    auto gt = readGroundTruth(sudokusGtPath);

    std::cout << "OpenCV Version: " << cv::getVersionString() << std::endl;
    std::cout << "Found " << gt.size() << " ground truth entries" << std::endl;

    int sudokuNum = 0;
    for (const auto& item : gt) {
        std::cout << "Sudoku " << sudokuNum << std::endl;

        processSudoku(item.first, item.second);

        sudokuNum++;
    }
}
