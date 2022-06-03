#include "CellClassifier.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
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

CellClassifier::CellClassifier(const std::string& classifierPath)
  : pimpl{ std::make_unique<Impl>(classifierPath) }
{
}

CellClassifier::CellClassifier(CellClassifier&&) noexcept = default;

CellClassifier::~CellClassifier() = default;

CellClassifier&
CellClassifier::operator=(CellClassifier&&) noexcept = default;

std::vector<int>
CellClassifier::classify(const std::vector<cv::Mat>& patches) const
{
    return pimpl->classify(patches);
}
int

CellClassifier::classify(const cv::Mat& patch) const
{
    return pimpl->classify(patch);
}
std::vector<int>
CellClassifier::classify(const cv::Mat& frame, const std::vector<Quad>& cellBounds) const
{
    const int pad = 3;
    const int patchSize = 64;
    const float padNeg = -pad;
    const float padPos = patchSize + pad;
    const cv::Size croppedSize(patchSize, patchSize);
    std::array<cv::Point2f, 4> to{
        cv::Point2f(padNeg, padNeg),
        cv::Point2f(padPos, padNeg),
        cv::Point2f(padPos, padPos),
        cv::Point2f(padNeg, padPos),
    };

    std::vector<cv::Mat> patches(cellBounds.size());
    for (size_t i = 0; i < cellBounds.size(); i++) {
        const auto& corners = cellBounds.at(i).corners;
        const std::array<cv::Point2f, 4> from{
            static_cast<cv::Point2f>(corners.at(0)),
            static_cast<cv::Point2f>(corners.at(1)),
            static_cast<cv::Point2f>(corners.at(2)),
            static_cast<cv::Point2f>(corners.at(3)),
        };

        const auto m = cv::getPerspectiveTransform(from, to);
        cv::warpPerspective(frame, patches.at(i), m, croppedSize, cv::INTER_AREA);
    }

    return classify(patches);
}
