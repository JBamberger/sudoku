#include "DigitClassifier.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <fstream>

#include <assert.h>

class DigitClassifier::Impl {
	constexpr static int IMAGE_SIZE = 20;
	const std::string SVM_PATH = "SVM_DIGITS.yml";
	const std::string TRAINING_PATH = "../../../share/digits.png";

	cv::Ptr<cv::ml::SVM> svm;
	cv::HOGDescriptor hogDescriptor;

public:
	Impl() {
		hogDescriptor = HOGDescriptor{
		Size(20, 20), //winSize
		Size(8, 8), //blocksize
		Size(4, 4), //blockStride,
		Size(8, 8), //cellSize,
		9, //nbins,
		1, //derivAper,
		-1, //winSigma,
		0, //histogramNormType,
		0.2, //L2HysThresh,
		0,//gammal correction,
		64,//nlevels=64
		1 };
		svm = loadOrTrain();
	}

	~Impl() {
	}

	int classify(const cv::Mat& img) {
		std::vector<float> features = preprocessImage(img);

		const size_t feat_cnt = features.size();
		cv::Mat preprocessedImg(1, feat_cnt, CV_32FC1);
		cv::Mat response;

		for (size_t j = 0; j < feat_cnt; j++) {
			preprocessedImg.at<float>(0, j) = features.at(j);
		}

		try {
			svm->predict(preprocessedImg, response);
			return response.at<float>(0, 0);
		}
		catch (const cv::Exception& e) {
			std::cout << e.what() << std::endl;
			throw e;
		}
	}

private:
	cv::Ptr<cv::ml::SVM> loadOrTrain() {
		try {
			std::cout << "Trying to load SVM from disk." << std::endl;

			cv::Ptr<cv::ml::SVM> svm = Algorithm::load<ml::SVM>(SVM_PATH);

			std::cout << "Loading finished." << std::endl;
			return svm;
		}
		catch (...) {
			std::cout << "Could not load SVM from disk. Begin training." << std::endl;

			// Load the training data from the image file
			cv::Mat digits;
			std::vector<int> labels;
			std::tie(digits, labels) = loadTrainingData(TRAINING_PATH);


			// SVM initialization
			cv::Ptr<cv::ml::SVM> svm = ml::SVM::create();
			svm->setType(ml::SVM::C_SVC);
			svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 500, 1e-6));

			//svm->setKernel(ml::SVM::LINEAR);
			svm->setKernel(ml::SVM::RBF);
			svm->setGamma(0.5); // 0.5
			svm->setC(12.5); // 12.5

			// SVM training
			try {
				svm->train(digits, ml::SampleTypes::ROW_SAMPLE, labels);
			}
			catch (const cv::Exception& e) {
				std::cout << e.what() << std::endl;
				throw e;
			}

			// Persist SVM for the next invokation.
			svm->save(SVM_PATH);

			std::cout << "Training finished." << std::endl;
			return svm;
		}
	}


	/**
	* This function deskews an input image of size 20x20 grayscale and then computes HoG features.
	*/
	std::vector<float> preprocessImage(const cv::Mat& image) {
		assert(image.size().height == 20);
		assert(image.size().width == 20);
		assert(image.type() == CV_8U);

		// Deskew image if necessary
		const cv::Moments& moments = cv::moments(image);
		cv::Mat imgOut = cv::Mat::zeros(image.rows, image.cols, image.type());
		if (cv::abs(moments.mu02 < 1e-2)) {
			imgOut = image;
		}
		else {
			const double skew = moments.mu11 / moments.mu02;
			cv::Mat warpMat = (cv::Mat_<double>(2, 3) << 1, skew, -0.5 * IMAGE_SIZE * skew, 0, 1, 0);
			cv::warpAffine(image, imgOut, warpMat, imgOut.size(), WARP_INVERSE_MAP | INTER_LINEAR);
		}

		// compute HoG features
		std::vector<float> descriptors;
		hogDescriptor.compute(imgOut, descriptors);

		return descriptors;
	}



	//    Mat small = Mat(20,20, CV_8U);
	//    cv::resize(image, small, Size(20,20));

	//Mat output = Mat(28, 28, CV_8U);
	//cv::resize(image, output, Size(28, 28));
	//output = output.reshape(0, 1);
	//output.convertTo(output, CV_32FC1);
	////std::cout << output.size().width << "x" << output.size().height << std::endl;
	//return output;

/**
* This function loads and preprocessed the training data set which is
* stored in a grayscale image file with five rows of 100 images per
* class.
*/
	std::pair<cv::Mat, std::vector<int>> loadTrainingData(const std::string& path) {
		cv::Mat inputImage = cv::imread(path, IMREAD_GRAYSCALE);

		if (inputImage.data == nullptr) {
			std::cout << "Could not read training data image." << std::endl;
			throw std::exception("Could not read training data.");
		}

		std::vector<std::vector<float>> features;
		features.reserve(5000);
		std::vector<int> labels;
		labels.reserve(5000);

		for (int r = 0; r < inputImage.rows; r = r + IMAGE_SIZE) {
			for (int c = 0; c < inputImage.cols; c = c + IMAGE_SIZE) {
				const cv::Mat& digit = inputImage.colRange(c, c + IMAGE_SIZE).rowRange(r, r + IMAGE_SIZE);
				std::vector<float> preprocessedDigit = preprocessImage(digit);
				features.push_back(preprocessedDigit);
				labels.push_back(r / (5 * IMAGE_SIZE));
			}
		}

		assert(features.size() == 5000);
		assert(labels.size() == 5000);

		const size_t len = features.size();
		const size_t feat_cnt = features.at(0).size();

		int im_cnt = 0;
		cv::Mat images(len, feat_cnt, CV_32FC1);

		for (size_t i = 0; i < len; i++) {
			for (size_t j = 0; j < feat_cnt; j++) {
				images.at<float>(i, j) = features.at(i).at(j);
			}
		}

		return std::pair<cv::Mat, std::vector<int>>(images, labels);
	}
};


DigitClassifier::DigitClassifier() : pimpl{ new Impl } {}

DigitClassifier::~DigitClassifier() = default;

int DigitClassifier::classify(const cv::Mat& img) {
	return pimpl->classify(img);
}