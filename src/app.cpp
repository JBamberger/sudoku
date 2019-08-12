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

void drawLine(const cv::Vec2f& line, cv::Mat& img, const cv::Scalar& rgb = CV_RGB(0, 0, 255)) {
	cv::Point2i p1;
	cv::Point2i p2;
	if (line[1] != 0) {
		const float m = -1 / tan(line[1]);
		const float c = line[0] / sin(line[1]);
		const int w = img.size().width;
		p1 = cv::Point2i(0, static_cast<int>(c));
		p2 = cv::Point2i(w, static_cast<int>(m * w + c));
	}
	else {
		p1 = cv::Point2i(static_cast<int>(line[0]), 0);
		p2 = cv::Point2i(static_cast<int>(line[0]), img.size().height);
	}
	cv::line(img, p1, p2, rgb);
}

cv::Rect find_biggest_object(const cv::Mat& input) {
	double max_area = 0;
	int max_index = 0;

	std::vector< std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;

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

std::vector<cv::Point> find_corners(const cv::Mat& image, float thresh) {
	int blockSize = 3;
	int apertureSize = 3;
	double k = 0.04;

	std::vector<cv::Point> corners;
	cv::Mat harris_response = cv::Mat::zeros(image.size(), CV_32FC1);
	cv::cornerHarris(image, harris_response, blockSize, apertureSize, k);

	cv::Mat dst_norm;
	cv::normalize(harris_response, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, cv::Mat());

	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > thresh)
			{
				corners.push_back(cv::Point{ j,i });
			}
		}
	}

	return corners;
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

std::vector<cv::Vec2f>
discardLines(const std::vector<cv::Vec3f>& lines, const float minAngleDiff, const float minRadDiff) {
	std::vector<cv::Vec2f> maxLines;
	for (const auto& line1 : lines) {
		unsigned int wins = 0;
		for (const auto& line2 : lines) {
			float angleDiff = line1[1] - line2[1];
			float radDiff = line1[0] - line2[0];
			//            std::cout << angleDiff << " " << radDiff << std::endl;
			if (-minAngleDiff <= angleDiff && angleDiff <= minAngleDiff
				&& -minRadDiff <= radDiff && radDiff <= minRadDiff
				&& line1[2] < line2[2]) {
				continue;
			}
			wins++;
		}
		if (wins == lines.size()) {
			maxLines.emplace_back(line1[0], line1[1]);
		}
	}
	return maxLines;
}

int main(int argc, char* argv[]) {

	cv::Mat sudoku = cv::imread(R"(../../../share/sudoku-sk.jpg)", 0);
	cv::resize(sudoku, sudoku, cv::Size(), 0.5, 0.5);

	cv::Mat blurred;
	cv::Mat thresh;
	cv::Mat inverted;
	cv::Mat dilated;

	cv::GaussianBlur(sudoku, blurred, cv::Size(11, 11), 3);
	cv::adaptiveThreshold(blurred, thresh, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
	cv::bitwise_not(thresh, inverted);
	cv::dilate(inverted, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{ 5,5 }));

	//cv::imshow("image", sudoku);
	//cv::imshow("thresh", thresh);
	//cv::imshow("dilated", dilated);

	cv::Rect bbox = find_biggest_object(dilated);
	cv::rectangle(sudoku, bbox, cv::Scalar{ 0,255,0 });
	cv::imshow("bbox", sudoku);

	cv::Size dsize{ 512,512 };
	cv::Mat crop;
	cv::resize(inverted(bbox), crop, dsize);
	//cv::imshow("crop", crop);

	//cv::Mat threshEdges;
	//cv::Canny(thresh, threshEdges, 50, 200);
	//cv::imshow("threshedges", threshEdges);
	//cv::imshow("edges", edges);


	// center must fall into bbox
	std::vector<cv::Vec2f> h_lines;
	std::vector<cv::Vec2f> v_lines;
	std::tie(h_lines, v_lines) = find_lines(crop, CV_PI / 90);

	std::vector<cv::Point2f> corners;
	for (const auto& h : h_lines) {
		for (const auto& v : v_lines) {
			corners.push_back(intersect(v, h));
		}
	}

	if (h_lines.size() < 1 || v_lines.size() < 1) {
		std::cout << "Not enough lines detected!" << std::endl;
		return -1;
	}

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

	cv::Point2f upper_left = intersect(min_hline, min_vline);
	cv::Point2f upper_right = intersect(max_hline, min_vline);
	cv::Point2f lower_left = intersect(max_hline, max_vline);
	cv::Point2f lower_right = intersect(min_hline, max_vline);


	std::cout << min_hline << " " << max_hline << " " << min_vline << " " << max_vline << std::endl;
	std::cout << upper_left << upper_right << lower_left << lower_right << std::endl;
	std::cout << "Found " << h_lines.size() << " horizontal lines." << std::endl;
	std::cout << "Found " << v_lines.size() << " vertical lines." << std::endl;
	std::cout << "Found " << corners.size() << " corners." << std::endl;

	cv::Mat l_img = crop.clone();
	cv::cvtColor(l_img, l_img, CV_GRAY2BGR);

	for (const cv::Vec2f& line : h_lines) {
		drawLine(line, l_img, CV_RGB(0, 255, 0));
	}
	for (const auto& line : v_lines) {
		drawLine(line, l_img, CV_RGB(255, 0, 0));
	}
	for (const auto& corner : corners) {
		cv::circle(l_img, corner, 3, CV_RGB(255, 0, 0), 1);
	}
	drawLine(min_hline, l_img, CV_RGB(255, 0, 255));
	drawLine(max_hline, l_img, CV_RGB(255, 0, 255));
	drawLine(min_vline, l_img, CV_RGB(255, 0, 255));
	drawLine(max_vline, l_img, CV_RGB(255, 0, 255));
	cv::circle(l_img, upper_left, 4, CV_RGB(255, 0, 255), 3);
	cv::circle(l_img, upper_right, 4, CV_RGB(255, 0, 255), 3);
	cv::circle(l_img, lower_left, 4, CV_RGB(255, 0, 255), 3);
	cv::circle(l_img, lower_right, 4, CV_RGB(255, 0, 255), 3);
	cv::imshow("lines and corners", l_img);



	const float dx = static_cast<float>(bbox.width) / static_cast<float>(dsize.width);
	const float dy = static_cast<float>(bbox.height) / static_cast<float>(dsize.height);
	cv::Point2f ul_b = cv::Point2f(bbox.x + upper_left.x * dx, bbox.y + upper_left.y * dy);
	cv::Point2f ur_b = cv::Point2f(bbox.x + upper_right.x * dx, bbox.y + upper_right.y * dy);
	cv::Point2f ll_b = cv::Point2f(bbox.x + lower_left.x * dx, bbox.y + lower_left.y * dy);
	cv::Point2f lr_b = cv::Point2f(bbox.x + lower_right.x * dx, bbox.y + lower_right.y * dy);
	std::cout << ul_b << ur_b << ll_b << lr_b << std::endl;

	cv::Mat s1;
	cv::cvtColor(sudoku, s1, CV_GRAY2BGR);
	cv::circle(s1, ul_b, 4, CV_RGB(255, 0, 255), 3);
	cv::circle(s1, ur_b, 4, CV_RGB(255, 0, 255), 3);
	cv::circle(s1, ll_b, 4, CV_RGB(255, 0, 255), 3);
	cv::circle(s1, lr_b, 4, CV_RGB(255, 0, 255), 3);
	cv::imshow("points", s1);

	std::vector<cv::Point2f> src;
	src.push_back(ul_b);
	src.push_back(ur_b);
	src.push_back(lr_b);
	src.push_back(ll_b);

	const int cell_size = 28;
	const int div_size = 2;
	const int pcell_size = cell_size + 2 * div_size;
	const cv::Size warp_size = { 9 * pcell_size, 9 * pcell_size };
	std::vector<cv::Point2f> dst;
	dst.push_back(cv::Point2i{ 0,0 });
	dst.push_back(cv::Point2i{ warp_size.width, 0 });
	dst.push_back(cv::Point2i{ 0, warp_size.height });
	dst.push_back(cv::Point2i{ warp_size.width, warp_size.height });
	cv::Mat warped;
	cv::warpPerspective(sudoku, warped, cv::getPerspectiveTransform(src, dst), warp_size);

	cv::imshow("warped", warped);

	std::vector<cv::Mat> cells;
	cells.reserve(81);
	for (int row = 0; row < 9; row++) {
		for (int col = 0; col < 9; col++) {
			cv::Rect cellRect(div_size + col * pcell_size, div_size + row * pcell_size, cell_size, cell_size);
			cells.push_back(warped(cellRect));
		}
	}

	for (const auto& cell : cells) {
		cv::imshow("cell", cell);
		cv::waitKey();
	}





	//float minAngleDiff = 0;
	//float minRadDiff = 0;
	//auto draw = [&]() {
	//    cv::Mat sudoku2 = cv::Mat(sudoku);
	//    cv::cvtColor(sudoku2, sudoku2, CV_GRAY2RGB);
	//    std::vector<cv::Vec2f> maxLines = discardLines(lines, minAngleDiff, minRadDiff);
	//    std::cout << minAngleDiff << std::endl;
	//    for (const auto& line : maxLines) {
	//        drawLine(line, sudoku2, CV_RGB(0, 255, 0));
	//    }
	//    cv::imshow("lines", sudoku2);
	//};

	//draw();



	//using TrackbarAction = std::function<void(int)>;
	//TrackbarAction angleChanged = [&](int value) {
	//    minAngleDiff = ((float)value) * (CV_PI / 180);
	//    draw();
	//};

	//TrackbarAction radChanged = [&](int value) {
	//    minRadDiff = value;
	//    draw();
	//};

	//cv::TrackbarCallback trackbarCallback = [](int pos, void* userdata) {
	//    (*(TrackbarAction*)userdata)(pos);
	//};

	//int angleVal = 0;
	//int radVal = 0;
	//cv::createTrackbar("angle", "lines", &angleVal, 360, trackbarCallback, (void*)& angleChanged);
	//cv::createTrackbar("radius", "lines", &radVal, 1000, trackbarCallback, (void*)& radChanged);


	cv::waitKey();

	return EXIT_SUCCESS;
}
