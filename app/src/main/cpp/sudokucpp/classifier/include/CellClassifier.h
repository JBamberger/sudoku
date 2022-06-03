#ifndef CELLCLASSIFIER_H
#define CELLCLASSIFIER_H

#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>

#include <geometry.h>

class CellClassifier
{
    struct Impl;
    std::unique_ptr<Impl> pimpl;

  public:
    explicit CellClassifier(const std::string& classifierPath);
    CellClassifier(CellClassifier&&) noexcept;
    CellClassifier(const CellClassifier&) = delete;
    ~CellClassifier();
    CellClassifier& operator=(CellClassifier&&) noexcept;
    CellClassifier& operator=(const CellClassifier&) = delete;


    [[nodiscard]] std::vector<int> classify(const std::vector<cv::Mat>& patches) const;
    [[nodiscard]] int classify(const cv::Mat& patch) const;
    [[nodiscard]] std::vector<int> classify(const cv::Mat& frame, const std::vector<Quad>& cellBounds) const;
};

#endif // CELLCLASSIFIER_H
