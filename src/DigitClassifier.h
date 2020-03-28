#pragma once

#include "opencv2/core/core.hpp"
#include <memory>

using namespace cv;

class DigitClassifier
{
    class Impl;
    std::unique_ptr<Impl> pimpl;

  public:
    DigitClassifier();
    ~DigitClassifier();
    int classify(const cv::Mat& img);
};