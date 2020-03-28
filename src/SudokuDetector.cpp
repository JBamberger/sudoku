#include "SudokuDetector.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class SudokuDetector::Impl
{
    const int cell_size = 28;
    const int div_size = 2;
    const int pcell_size = cell_size + 2 * div_size;
    const cv::Size warp_size = cv::Size(9 * pcell_size, 9 * pcell_size);
    const cv::Size dsize{ 512, 512 };

  public:
    explicit Impl() = default;
    ~Impl() = default;

    /**
     * This is the interface function which performs all steps of the sudoku
     * detection.
     */
    Sudoku detect_sudoku(const cv::Mat& input)
    {
        Sudoku sudoku{ input };
        cv::Mat inverted;
        preprocess_image(sudoku, inverted);
        detect_sudoku_corners(sudoku, inverted);
        compute_unwarp_transform(sudoku);
        get_cells(sudoku);
        get_cell_contents(sudoku);
        return sudoku;
    }

  private:
    /**
     * Creates a slightly blurred inverted binarized input image.
     */
    static void preprocess_image(Sudoku& sudoku, cv::Mat& inverted)
    {
        // preprocess input image for later steps: blurring, thresholding, inverting
        // and dilating
        cv::Mat blurred;
        cv::GaussianBlur(sudoku.input, blurred, cv::Size(11, 11), 3);
        cv::adaptiveThreshold(blurred, inverted, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
        cv::bitwise_not(inverted, inverted);
    }

    /**
     * The goal of this function is to find a rectangle which enclosed
     * the largest object in the image
     */
    static cv::Rect find_rough_crop_region(const cv::Mat& inverted)
    {
        // find the biggest object in the image and transform it to a fixed size
        cv::Mat dilated;
        cv::dilate(inverted, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{ 5, 5 }));

        double max_area = 0;
        int max_index = 0;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(dilated, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            double a = cv::contourArea(contours[i], false);
            if (a > max_area) {
                max_area = a;
                max_index = i;
            }
        }

        return cv::boundingRect(contours[max_index]);
    }

    /**
     * This function performs non-maximum suppression for lines by eliminating all
     * lines which are closer than min_dist.
     */
    static std::vector<cv::Vec2f> nms_line(std::vector<cv::Vec3f>& lines, const float min_dist = 20.f)
    {
        std::vector<cv::Vec2f> out_lines;

        std::sort(lines.begin(), lines.end(), [](const auto& a, const auto& b) { return a[2] > b[2]; });

        for (size_t i = 0; i < lines.size(); i++) {
            bool add = true;
            const auto& l1 = lines.at(i);

            for (size_t j = i + 1; j < lines.size(); j++) {
                if (std::abs(l1[0] - lines.at(j)[0]) < min_dist) {
                    add = false;
                }
            }
            if (add) {
                out_lines.push_back({ l1[0], l1[1] });
            }
        }
        return out_lines;
    }

    /**
     * Function computes a set of horizontal and vertical lines in the given image.
     * The threshold theta specifies the allowed deviation from the horizontal /
     * vertical axis.
     */
    static std::pair<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> find_lines(const cv::Mat& image, double theta)
    {
        static constexpr double PI2 = CV_PI / 2;

        const double distance_resolution = 1;        // distance resolution in pixels
        const double angle_resolution = CV_PI / 360; // angle resolution in radians
        const int support_threshold = 450;           // min. number of supporting pixels

        // cv::HoughLines(image, h_lines, distance_resolution, angle_resolution,
        // support_threshold, 0, 0, 0, theta); cv::HoughLines(image, h_lines,
        // distance_resolution, angle_resolution, support_threshold, 0, 0, CV_PI -
        // theta, CV_PI); cv::HoughLines(image, v_lines, distance_resolution,
        // angle_resolution, support_threshold, 0, 0, PI2 - theta, PI2 + theta);

        std::vector<cv::Vec3f> lines;
        cv::HoughLines(image, lines, distance_resolution, angle_resolution, support_threshold, 0, 0, 0, CV_PI);

        std::vector<cv::Vec3f> h_lines_tmp;
        std::vector<cv::Vec3f> v_lines_tmp;

        for (const auto& line : lines) {
            if ((0 <= line[1] && line[1] <= theta) || ((CV_PI - theta) <= line[1] && line[1] <= CV_PI)) {
                h_lines_tmp.push_back(line);
            } else if (PI2 - theta <= line[1] && line[1] <= PI2 + theta) {
                v_lines_tmp.push_back(line);
            } else {
                std::cout << "could not match line" << std::endl;
            }
        }

        // remove non maximum lines
        std::vector<cv::Vec2f> h_lines = nms_line(h_lines_tmp);
        std::vector<cv::Vec2f> v_lines = nms_line(v_lines_tmp);

        return std::pair<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>>(h_lines, v_lines);
    }

    /**
     * Function which computes the intersection of two lines given as angle and
     * distance from origin.
     */
    static cv::Point2f intersect(const cv::Vec2f& l1, const cv::Vec2f& l2, const double epsilon = 1e-15)
    {

        cv::Vec2d n{ std::cos(l1[1]), std::sin(l1[1]) };
        cv::Vec2d m{ std::cos(l2[1]), std::sin(l2[1]) };
        n = n / cv::norm(n);
        m = m / cv::norm(m);

        const double n2m1 = n[1] * m[0];
        const double n1m2 = n[0] * m[1];

        if (std::abs(n2m1 - n1m2) < epsilon) {
            throw std::exception("Could not compute intersection of parallel lines");
        }

        const double x = (l2[0] * n[1] - l1[0] * m[1]) / (n2m1 - n1m2);
        const double y = (l2[0] * n[0] - l1[0] * m[0]) / (n1m2 - n2m1);

        return cv::Point2f{ static_cast<float>(x), static_cast<float>(y) };
    }

    /**
     * This function computes the minimal and maximal lines in horizontal or
     * vertical direction. The angleFunction is used to compute the projected
     * distance of the line.
     */
    static std::pair<cv::Vec2f, cv::Vec2f> find_min_max_line_pair(std::vector<cv::Vec2f> lines,
                                                                  float angleFunction(float))
    {
        cv::Vec2f min_line = lines.at(0);
        cv::Vec2f max_line = lines.at(0);
        for (const auto& line : lines) {
            const float dx = angleFunction(line[1]) * line[0];
            if (dx > angleFunction(max_line[1]) * max_line[0]) {
                max_line = line;
            }
            if (dx < angleFunction(min_line[1]) * min_line[0]) {
                min_line = line;
            }
        }
        return std::pair<cv::Vec2f, cv::Vec2f>(min_line, max_line);
    }

    /**
     * This function detects the four corners of the sudoku.
     */
    void detect_sudoku_corners(Sudoku& sudoku, const cv::Mat& inverted)
    {
        // crop down to the approximate sudoku size and scale it to a fixed size
        sudoku.bbox = find_rough_crop_region(inverted);
        cv::Mat crop;
        cv::resize(inverted(sudoku.bbox), crop, dsize);

        // find all horizontal and vertical lines
        std::vector<cv::Vec2f> h_lines;
        std::vector<cv::Vec2f> v_lines;
        std::tie(h_lines, v_lines) = find_lines(crop, CV_PI / 90);
        if (h_lines.empty() || v_lines.empty())
            throw std::exception("Not enough lines detected.");

        // compute the bounding lines
        cv::Vec2f min_hline, max_hline, min_vline, max_vline;
        std::tie(min_hline, max_hline) = find_min_max_line_pair(h_lines, std::cos);
        std::tie(min_vline, max_vline) = find_min_max_line_pair(v_lines, std::sin);

        // compute the corners of the sudoku in the original image
        const cv::Point2f p{ static_cast<float>(sudoku.bbox.x), static_cast<float>(sudoku.bbox.y) };
        const cv::Point2f delta{ static_cast<float>(sudoku.bbox.width) / static_cast<float>(dsize.width),
                                 static_cast<float>(sudoku.bbox.height) / static_cast<float>(dsize.height) };

        auto uncrop_coords = [p, delta](const cv::Point2f& b) {
            return cv::Point2f(p.x + b.x * delta.x, p.y + b.y * delta.y);
        };

        sudoku.corners = Quad{ uncrop_coords(intersect(min_hline, min_vline)),
                               uncrop_coords(intersect(max_hline, min_vline)),
                               uncrop_coords(intersect(max_hline, max_vline)),
                               uncrop_coords(intersect(min_hline, max_vline)) };
    }

    /**
     * This function computes the aligned sudoku image and the transformations from
     * and to transformed space.
     */
    void compute_unwarp_transform(Sudoku& sudoku)
    {
        Quad destCorners{ cv::Point2i{ 0, 0 },
                          cv::Point2i{ warp_size.width, 0 },
                          cv::Point2i{ warp_size.width, warp_size.height },
                          cv::Point2i{ 0, warp_size.height } };

        // compute a perspective transformation to align the sudoku
        sudoku.warpMap = cv::getPerspectiveTransform(sudoku.corners.asVec(), destCorners.asVec());
        sudoku.unwarpMap = cv::getPerspectiveTransform(destCorners.asVec(), sudoku.corners.asVec());
        cv::warpPerspective(sudoku.input, sudoku.aligned, sudoku.warpMap, warp_size);
    }

    /**
     * This function computes the location of all sudoku cells.
     */
    void get_cells(Sudoku& sudoku)
    {
        sudoku.cells.reserve(81);
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                cv::Rect2i cellRect(div_size + col * pcell_size, div_size + row * pcell_size, cell_size, cell_size);
                sudoku.cells.push_back(cellRect);
            }
        }
    }

    static cv::Mat keep_largest_blob(const cv::Mat& in)
    {
        //    int count = 0;
        int max = -1;
        cv::Point2i maxPt;
        cv::Mat tmp = in.clone();
        cv::bitwise_not(tmp, tmp);

        for (int row = 0; row < tmp.rows; row++) {
            for (int col = 0; col < tmp.cols; col++) {
                if (tmp.at<uint8_t>(row, col) < 128)
                    continue; // skip processed and background pixels
                int area = cv::floodFill(tmp, cv::Point2i(col, row), 64);
                if (area <= max)
                    continue; // keep only the largest blob
                maxPt = cv::Point2i(col, row);
                max = area;
            }
        }
        tmp = 128 + in.clone() * 0.5;
        int area = cv::floodFill(tmp, maxPt, 0);

        cv::threshold(tmp, tmp, 64, 255, cv::THRESH_BINARY);

        if (max == -1)
            return in.clone();
        else
            return tmp;
    }

    void get_cell_contents(Sudoku& sudoku)
    {
        cv::Mat img;
        cv::Mat threshed;
        cv::GaussianBlur(sudoku.aligned, img, { 5, 5 }, 1);
        cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
        cv::medianBlur(img, img, 3);

        sudoku.cell_contents.reserve(81);
        for (const auto& cell : sudoku.cells) {
            cv::Mat t_cell = img(cell);

            const int t = 4;
            const int nnz_thresh = 20;
            const int s = cell_size * cell_size;
            for (int i = 0; i < t; i++) {
                if (s - cv::countNonZero(t_cell.row(i)) > nnz_thresh)
                    t_cell.row(i) = 255;
                if (s - cv::countNonZero(t_cell.row(cell_size - i - 1)) > nnz_thresh)
                    t_cell.row(cell_size - i - 1) = 255;
                if (s - cv::countNonZero(t_cell.col(i)) > nnz_thresh)
                    t_cell.col(i) = 255;
                if (s - cv::countNonZero(t_cell.col(cell_size - i - 1)) > nnz_thresh)
                    t_cell.col(cell_size - i - 1) = 255;
            }

            cv::Mat out = keep_largest_blob(t_cell);

            cv::Mat output_cell = cv::Mat(20, 20, CV_8U);
            cv::resize(out, output_cell, cv::Size(20, 20));
            output_cell = cv::Scalar::all(255) - output_cell;

            // const int C = 20 * 20;
            // std::cout << output_cell << (cv::countNonZero(output_cell) > 0.08 * C) <<
            // std::endl; cv::Mat h; cv::hconcat(t_cell, out, h); cv::imshow("cell",
            // output_cell); cv::waitKey();
            sudoku.cell_contents.push_back(output_cell);
        }
    }
};

SudokuDetector::SudokuDetector()
  : pimpl(std::make_unique<Impl>())
{}

SudokuDetector::~SudokuDetector() = default;

Sudoku
SudokuDetector::detect_sudoku(const cv::Mat& input)
{
    return pimpl->detect_sudoku(input);
}
