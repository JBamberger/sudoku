#ifndef CELLCLASSIFIER_H
#define CELLCLASSIFIER_H

#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>

class CellClassifier
{
    struct Impl;
    std::unique_ptr<Impl> pimpl;

  public:
    explicit CellClassifier();
    CellClassifier(CellClassifier&&) noexcept;
    CellClassifier(const CellClassifier&) = delete;
    ~CellClassifier();
    CellClassifier& operator=(CellClassifier&&) noexcept;
    CellClassifier& operator=(const CellClassifier&) = delete;

    std::vector<int> classify(const std::vector<cv::Mat>& patches);
    int classify(const cv::Mat& patch);
};

#endif // CELLCLASSIFIER_H
