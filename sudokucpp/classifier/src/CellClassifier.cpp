#include "CellClassifier.h"

#include <opencv2/dnn.hpp>
#include <string>

struct CellClassifier::Impl
{
    cv::dnn::Net net;

    explicit Impl(const std::string& modelPath)
      : net(cv::dnn::readNet(modelPath, "", ""))
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        // net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }

    std::vector<int> classify(const std::vector<cv::Mat>& patches)
    {
        // Preprocess
        cv::Mat input;
        cv::dnn::blobFromImages(patches, input, 1.0 / 255.0, cv::Size(), 128.0);
        cv::divide(input, 0.5, input);

        // Classify
        net.setInput(input);
        auto output = net.forward();

        std::vector<int> classes;
        classes.reserve(patches.size());
        for (int i = 0; i < output.rows; i++) {
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(output.row(i), nullptr, &confidence, nullptr, &classIdPoint);

            classes.push_back(classIdPoint.x);
        }

        return classes;
    }

    int classify(const cv::Mat& patch)
    {
        assert(patch.rows == 64 && patch.cols == 64);

        // Preprocess
        cv::Mat input;
        cv::dnn::blobFromImage(patch, input, 1.0 / 255.0, cv::Size(), 128.0);
        cv::divide(input, 0.5, input);

        // Classify
        net.setInput(input);
        auto output = net.forward();

        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);

        return classIdPoint.x;
    }
};

CellClassifier::CellClassifier()
  : pimpl{ std::make_unique<Impl>("D:/dev/sudoku/share/digit_classifier_ts.onnx") }
{}

CellClassifier::CellClassifier(CellClassifier&&) noexcept = default;

CellClassifier::~CellClassifier() = default;

CellClassifier&
CellClassifier::operator=(CellClassifier&&) noexcept = default;

std::vector<int>
CellClassifier::classify(const std::vector<cv::Mat>& patches)
{
    return pimpl->classify(patches);
}
int

CellClassifier::classify(const cv::Mat& patch)
{
    return pimpl->classify(patch);
}
