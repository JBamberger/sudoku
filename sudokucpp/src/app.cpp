#include "SudokuDetector.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>

int
main()
{
    //    auto detector = SudokuDetector();
    //
    //    cv::Mat sudokuImg = cv::imread(R"(../share/sudoku-sk.jpg)", cv::IMREAD_COLOR);
    //
    //    auto detection = detector.detect(sudokuImg);
    //
    //    auto canvas = sudokuImg.clone();
    //

    std::cout << "Hello World" << std::endl << "OpenCV Version: " << cv::getVersionString() << std::endl;
    std::cout << "Torch Version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH
              << std::endl;
}
