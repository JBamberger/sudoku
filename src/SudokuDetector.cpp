#include "SudokuDetector.h"

#include <opencv2/imgproc.hpp>
#include <iostream>   


class SudokuDetector::Impl {

	cv::Mat input;
	cv::Mat blurred;
	cv::Mat thresh;
	cv::Mat inverted;
	cv::Mat dilated;
	cv::Mat crop;
	cv::Mat warped;

	Quad corners;
	cv::Mat warpMap;
	cv::Mat unwarpMap;


	const int cell_size = 28;
	const int div_size = 2;
	const int pcell_size = cell_size + 2 * div_size;
	const cv::Size warp_size = { 9 * pcell_size, 9 * pcell_size };

	const cv::Size dsize{ 512,512 };

public:
	Impl(cv::Mat image) : input{ image } {}

	void detect() {

		// preprocess input image for later steps: blurring, thresholding, inverting and dilating
		cv::GaussianBlur(input, blurred, cv::Size(11, 11), 3);
		cv::adaptiveThreshold(blurred, thresh, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
		cv::bitwise_not(thresh, inverted);
		cv::dilate(inverted, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{ 5,5 }));

		// find the biggest object in the image and transform it to a fixed size
		cv::Rect bbox = find_biggest_object(dilated);
		cv::resize(inverted(bbox), crop, dsize);

		// find all horizontal and vertical lines
		std::vector<cv::Vec2f> h_lines;
		std::vector<cv::Vec2f> v_lines;
		std::tie(h_lines, v_lines) = find_lines(crop, CV_PI / 90);
		if (h_lines.size() < 1 || v_lines.size() < 1) throw std::exception("Not enough lines detected.");

		// compute the bounding lines
		cv::Vec2f min_hline = h_lines.at(0);
		cv::Vec2f max_hline = h_lines.at(0);
		cv::Vec2f min_vline = v_lines.at(0);
		cv::Vec2f max_vline = v_lines.at(0);
		for (const auto& line : h_lines) {
			const float dx = std::cos(line[1]) * line[0];
			if (dx > std::cos(max_hline[1]) * max_hline[0]) {
				max_hline = line;
			}
			if (dx < std::cos(min_hline[1]) * min_hline[0]) {
				min_hline = line;
			}
		}
		for (const auto& line : v_lines) {
			const float dy = std::sin(line[1]) * line[0];
			if (dy > std::sin(max_vline[1]) * max_vline[0]) {
				max_vline = line;
			}
			if (dy < std::sin(min_vline[1]) * min_vline[0]) {
				min_vline = line;
			}
		}
		// compute the corners of the sudoku in warped coordinates
		Quad warp_corns{
		intersect(min_hline, min_vline),
		intersect(max_hline, min_vline),
		intersect(max_hline, max_vline),
		intersect(min_hline, max_vline)
		};

		// compute the corners of the sudoku in the original image
		const float dx = static_cast<float>(bbox.width) / static_cast<float>(dsize.width);
		const float dy = static_cast<float>(bbox.height) / static_cast<float>(dsize.height);

		corners = Quad{
		cv::Point2f(bbox.x + warp_corns.tl.x * dx, bbox.y + warp_corns.tl.y * dy),
		cv::Point2f(bbox.x + warp_corns.tr.x * dx, bbox.y + warp_corns.tr.y * dy),
		cv::Point2f(bbox.x + warp_corns.br.x * dx, bbox.y + warp_corns.br.y * dy),
		cv::Point2f(bbox.x + warp_corns.bl.x * dx, bbox.y + warp_corns.bl.y * dy)
		};
		Quad destCorners{
		cv::Point2i{ 0,0 },
		cv::Point2i{ warp_size.width, 0 },
		cv::Point2i{ warp_size.width, warp_size.height },
		cv::Point2i{ 0, warp_size.height }
		};

		// compute a perspective transformation to align the sudoku
		warpMap = cv::getPerspectiveTransform(corners.asVec(), destCorners.asVec());
		unwarpMap = cv::getPerspectiveTransform(destCorners.asVec(), corners.asVec());
		cv::warpPerspective(blurred, warped, warpMap, warp_size);
	}

	Sudoku get_result() {
		return Sudoku{ warped, corners, get_cells(), warpMap, unwarpMap };
	}



private:
	cv::Rect find_biggest_object(const cv::Mat& input) {
		double max_area = 0;
		int max_index = 0;

		std::vector< std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;

		cv::findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		for (int i = 0; i < contours.size(); i++) {
			double a = cv::contourArea(contours[i], false);
			if (a > max_area) {
				max_area = a;
				max_index = i;
			}
		}

		return cv::boundingRect(contours[max_index]);
	}

	std::pair<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> find_lines(const cv::Mat& image, double theta) {
		static constexpr double PI2 = CV_PI / 2;

		const double distance_resolution = 1; // distance resolution in pixels
		const double angle_resolution = CV_PI / 360; // angle resolution in radians
		const int support_threshold = 450; // min. number of supporting pixels

		//cv::HoughLines(image, h_lines, distance_resolution, angle_resolution, support_threshold, 0, 0, 0, theta);
		//cv::HoughLines(image, h_lines, distance_resolution, angle_resolution, support_threshold, 0, 0, CV_PI - theta, CV_PI);
		//cv::HoughLines(image, v_lines, distance_resolution, angle_resolution, support_threshold, 0, 0, PI2 - theta, PI2 + theta);

		std::vector<cv::Vec3f> lines;
		cv::HoughLines(image, lines, distance_resolution, angle_resolution, support_threshold, 0, 0, 0, CV_PI);

		std::vector<cv::Vec3f> h_lines_tmp;
		std::vector<cv::Vec3f> v_lines_tmp;

		for (const auto& line : lines) {
			if ((0 <= line[1] && line[1] <= theta) || ((CV_PI - theta) <= line[1] && line[1] <= CV_PI)) {
				h_lines_tmp.push_back(line);
			}
			else if (PI2 - theta <= line[1] && line[1] <= PI2 + theta) {
				v_lines_tmp.push_back(line);
			}
			else {
				std::cout << "could not match line" << std::endl;
			}
		}
		std::vector<cv::Vec2f> h_lines = nms_line(h_lines_tmp);
		std::vector<cv::Vec2f> v_lines = nms_line(v_lines_tmp);

		// perform nms

		return std::pair<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>>(h_lines, v_lines);
	}

	cv::Point2f intersect(const cv::Vec2f& l1, const cv::Vec2f& l2, const double epsilon = 1e-15) {

		cv::Vec2d n{ std::cos(l1[1]), std::sin(l1[1]) };
		cv::Vec2d m{ std::cos(l2[1]), std::sin(l2[1]) };
		n = n / cv::norm(n);
		m = m / cv::norm(m);

		const double n2m1 = n[1] * m[0];
		const double n1m2 = n[0] * m[1];

		if (std::abs(n2m1 - n1m2) < epsilon) {
			throw new std::exception("Could not compute intersection of parallel lines");
		}

		const double x = (l2[0] * n[1] - l1[0] * m[1]) / (n2m1 - n1m2);
		const double y = (l2[0] * n[0] - l1[0] * m[0]) / (n1m2 - n2m1);

		return cv::Point2f{ static_cast<float>(x),static_cast<float>(y) };
	}

	std::vector<cv::Rect2i> get_cells() {
		std::vector<cv::Rect2i> cells;
		cells.reserve(81);
		for (int row = 0; row < 9; row++) {
			for (int col = 0; col < 9; col++) {
				cv::Rect2i cellRect(div_size + col * pcell_size, div_size + row * pcell_size, cell_size, cell_size);
				cells.push_back(cellRect);
			}
		}
		return cells;
	}

	std::vector<cv::Vec2f> nms_line(std::vector<cv::Vec3f>& lines, const float min_dist = 20.f) {
		std::vector<cv::Vec2f> out_lines;

		std::sort(lines.begin(), lines.end(), [](const auto& a, const auto& b) {
			return a[2] > b[2];
			});

		for (size_t i = 0; i < lines.size(); i++) {
			bool add = true;
			const auto& l1 = lines.at(i);

			for (size_t j = i + 1; j < lines.size(); j++) {
				if (std::abs(l1[0] - lines.at(j)[0]) < min_dist) {
					add = false;
				}
			}
			if (add) {
				out_lines.push_back({ l1[0],l1[1] });
			}
		}
		return out_lines;
	}

};


SudokuDetector::SudokuDetector(cv::Mat image) : pimpl(new Impl(image))
{
}

SudokuDetector::~SudokuDetector() = default;

Sudoku SudokuDetector::get_result()
{

	pimpl->detect();

	return pimpl->get_result();
}
