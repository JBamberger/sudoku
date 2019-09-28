#include "app.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>

#include "DigitClassifier.h"
#include "SudokuDetector.h"

cv::Mat keep_largest_blob(const cv::Mat& in) {
	int count = 0;
	int max = -1;
	cv::Point2i maxPt;
	cv::Mat tmp = in.clone();
	cv::bitwise_not(tmp, tmp);

	for (int row = 0; row < tmp.rows; row++) {
		for (int col = 0; col < tmp.cols; col++) {
			if (tmp.at<uint8_t>(row, col) < 128) continue; // skip processed and background pixels
			int area = cv::floodFill(tmp, cv::Point2i(col, row), 64);
			if (area <= max) continue; // keep only the largest blob
			maxPt = cv::Point2i(col, row);
			max = area;
		}
	}
	tmp = 128 + in.clone() * 0.5;
	int area = cv::floodFill(tmp, maxPt, 0);

	cv::threshold(tmp, tmp, 64, 255, THRESH_BINARY);

	return tmp;
}


int main(int argc, char* argv[]) {

	cv::Mat sudokuImg = cv::imread(R"(../../../share/sudoku-sk.jpg)", 0);
	cv::resize(sudokuImg, sudokuImg, cv::Size(), 0.5, 0.5);

	DigitClassifier classifier = DigitClassifier();

	Sudoku sudoku = detect_sudoku(sudokuImg);

	const int cell_size = 28;
	std::vector<int> digits;
	digits.reserve(81);
	for (const auto& cell : sudoku.cells) {
		cv::Mat t_cell;
		cv::adaptiveThreshold(sudoku.aligned(cell), t_cell, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
		cv::medianBlur(t_cell, t_cell, 3);


		const int t = 4;
		const int nnz_thresh = 20;
		const int s = cell_size * cell_size;
		for (int i = 0; i < t; i++) {
			if (s - cv::countNonZero(t_cell.row(i)) > nnz_thresh) t_cell.row(i) = 255;
			if (s - cv::countNonZero(t_cell.row(cell_size - i - 1)) > nnz_thresh) t_cell.row(cell_size - i - 1) = 255;
			if (s - cv::countNonZero(t_cell.col(i)) > nnz_thresh) t_cell.col(i) = 255;
			if (s - cv::countNonZero(t_cell.col(cell_size - i - 1)) > nnz_thresh) t_cell.col(cell_size - i - 1) = 255;
		}

		cv::Mat out = keep_largest_blob(t_cell);

		//cv::Mat h;		cv::hconcat(t_cell, out, h);		cv::imshow("cell", h); cv::waitKey();

		int digit = -1;
		// invoke the classifier only if there is a minimum number of pixels set
		if (s - cv::countNonZero(t_cell) > 0.08 * s) {
			cv::Mat small = cv::Mat(20, 20, CV_8U);
			cv::resize(t_cell, small, cv::Size(20, 20));
			small = cv::Scalar::all(255) - small;
			digit = classifier.classify(small);
		}

		digits.push_back(digit);

		// visualize cell content and wait
		// cv::imshow("cell", t_cell); cv::waitKey();
		// std::cout << "digit=" << digit << ", count=" << (s - cv::countNonZero(t_cell)) << std::endl;
	}

	cv::Mat resultImg;
	cv::cvtColor(sudokuImg, resultImg, COLOR_GRAY2BGR);

	for (int i = 0; i < sudoku.cells.size(); i++) {
		//cv::imshow("cell", sudoku.aligned(sudoku.cells.at(i)));
		//cv::waitKey();

		if (digits.at(i) >= 0) {
			cv::Rect2i cell = sudoku.cells.at(i);

			cv::Mat warpedLocation = (cv::Mat_<double>(3, 1) <<
				(double)cell.x + cell.width * 0.1,
				(double)cell.y + cell.height * 0.9,
				1.f);
			cv::Mat u = sudoku.unwarpMap * warpedLocation;
			u = u / u.at<double>(2, 0);
			cv::Point2i location{ (int)u.at<double>(0,0), (int)u.at<double>(1,0) };

			cv::putText(resultImg, std::to_string(digits.at(i)), location,
				FONT_HERSHEY_COMPLEX_SMALL, 1, { 0, 255, 0 }, 1, CV_AA);
		}


		// std::cout << digits.at(i) << std::endl;
	}

	Quad c = sudoku.corners;
	cv::line(resultImg, c.tl, c.tr, { 0,255,0 });
	cv::line(resultImg, c.tr, c.br, { 0,255,0 });
	cv::line(resultImg, c.br, c.bl, { 0,255,0 });
	cv::line(resultImg, c.bl, c.tl, { 0,255,0 });
	cv::rectangle(resultImg, sudoku.bbox, { 0,0,255 });
	cv::imshow("Result", resultImg);
	cv::waitKey();

	return EXIT_SUCCESS;
}
