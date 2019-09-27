#pragma once

#include <opencv2/core/core.hpp>
#include <memory>
#include <vector>

struct Quad {
	cv::Point2f tl;
	cv::Point2f tr;
	cv::Point2f br;
	cv::Point2f bl;

	inline std::vector< cv::Point2f> asVec() {
		std::vector<cv::Point2f> out;
		out.push_back(tl);
		out.push_back(tr);
		out.push_back(br);
		out.push_back(bl);
		return out;
	}
};

struct Sudoku {
	// aligned (square) version of the sudoku
	cv::Mat aligned;

	// corner points in the original image
	Quad corners;

	// sudoku cells in row-major order
	std::vector<cv::Rect2i> cells;

	cv::Mat warpMap;   // from unwarped -> warped
	cv::Mat unwarpMap; // from warped -> unwarped
};

class SudokuDetector {
	class Impl;
	std::unique_ptr<Impl> pimpl;

public:
	SudokuDetector(cv::Mat image);
	~SudokuDetector();
	Sudoku get_result();
};