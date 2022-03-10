#include <filesystem>
#include <iostream>
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

int
main(int argc, const char* argv[])
{
    std::vector<fs::path> imagePaths{
        "D:/dev/sudoku/data/cells/0.jpg",  // 0: empty
        "D:/dev/sudoku/data/cells/71.jpg", // 0: garbage
        "D:/dev/sudoku/data/cells/37.jpg", // 1
        "D:/dev/sudoku/data/cells/13.jpg", // 2
        "D:/dev/sudoku/data/cells/35.jpg", // 3
        "D:/dev/sudoku/data/cells/62.jpg", // 4
        "D:/dev/sudoku/data/cells/45.jpg", // 5
        "D:/dev/sudoku/data/cells/58.jpg", // 6
        "D:/dev/sudoku/data/cells/59.jpg", // 7
        "D:/dev/sudoku/data/cells/1.jpg",  // 8
        "D:/dev/sudoku/data/cells/6.jpg",  // 9
    };

    std::vector<cv::Mat> images;
    for (const auto& imagePath : imagePaths) {
        cv::Mat cell = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        cv::Mat grayCell;
        cv::cvtColor(cell, grayCell, cv::COLOR_BGR2GRAY);
        images.push_back(grayCell);
    }

    //    torch::jit::script::Module module;
    //    try {
    //        module = torch::jit::load(argv[1]);
    //    } catch (const c10::Error& e) {
    //        std::cerr << "error loading the model\n";
    //        std::cerr << e.what() << std::endl;
    //        return -1;
    //    }

    const auto model = "D:/dev/sudoku/share/digit_classifier_ts.onnx";
    const auto config = "";
    const auto framework = "";

    cv::dnn::Net net = cv::dnn::readNet(model, config, framework);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    //    net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    for (const auto& grayCell : images) {

//        std::cout << "Type: " << grayCell.type() << std::endl;
        cv::Mat input;
        cv::dnn::blobFromImage(grayCell, input, 1.0 / 255.0, cv::Size(), 128.0);
        cv::divide(input, 0.5, input);

        net.setInput(input);
        cv::Mat output = net.forward();

        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);

        std::cout << classIdPoint.x << std::endl;
    }

    cv::Mat input;
    cv::dnn::blobFromImages(images, input, 1.0 / 255.0, cv::Size(), 128.0);
    cv::divide(input, 0.5, input);

    net.setInput(input);
    cv::Mat output = net.forward();

    std::cout << "#################" << std::endl;

    double min, max;
    cv::minMaxLoc(output, &min, &max, nullptr, nullptr);
    std::cout << min << " " << max << std::endl;

    for (int i = 0; i < output.rows; i++) {
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output.row(i), nullptr, &confidence, nullptr, &classIdPoint);
        std::cout << classIdPoint.x << " confidence: " << confidence << std::endl;
    }

    //    pad = 6
    //    _, pt = cv.threshold(gray_cell[pad:-pad, pad:-pad], 100, 255, cv.THRESH_BINARY_INV)
    //    cell_filled = np.count_nonzero(pt) > 100 and np.var(gray_cell) > 500

    //    bool cellFilled = true;
    //    int result = 0;
    //    if (cellFilled) {
    //        // to tensor
    //        // Normalize(0.5, 0.5)
    //
    //        auto results = module.forward(inputs);
    //    }
    //
    //    std::cout << "ok\n";
}