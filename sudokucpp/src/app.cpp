#include "SudokuDetector.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
//#include <torch/torch.h>
#include "config.h"
#include "utils.h"
#include <filesystem>

namespace fs = std::filesystem;

void
processSudoku(fs::path path, Quad gt_bbox)
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
    //    std::cout << "Torch Version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." <<
    //    TORCH_VERSION_PATCH
    //              << std::endl;

    std::cout << "Found " << gt.size() << " ground truth entries" << std::endl;

    for (const auto& item : gt) {
        processSudoku(item.first, item.second);
    }
}
