#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace cv;

class DigitClassifier
{
public:
	DigitClassifier();
	~DigitClassifier();
	int classify(cv::Mat img);

private:
	cv::Ptr<cv::ml::SVM> svm;
    void train();
    bool loadMNIST(const std::string &pic_filename, const std::string &label_filename, Mat &training_data, Mat &label_data);

    static Mat preprocessImage(Mat &image);
};