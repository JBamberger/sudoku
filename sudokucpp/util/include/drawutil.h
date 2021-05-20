#ifndef DRAWUTIL_H
#define DRAWUTIL_H

#include <array>
#include <opencv2/imgproc.hpp>
#include <vector>

template<typename T, size_t N>
std::vector<cv::Point>
asContour(const std::array<cv::Point_<T>, N>& points)
{
    std::vector<cv::Point> out;
    out.reserve(N);
    for (const auto& p : points) {
        out.push_back(static_cast<cv::Point>(p));
    }
    return out;
}

template<typename T>
void
drawOrientedRect(cv::Mat& canvas, const std::vector<cv::Point_<T>>& contour)
{
    assert(contour.size() == 4);

    const cv::Point& p1(contour.at(0));
    const cv::Point& p2(contour.at(1));
    const cv::Point& p3(contour.at(2));
    const cv::Point& p4(contour.at(3));
    cv::line(canvas, p1, p2, { 255, 0, 0 }, 2);
    cv::line(canvas, p2, p3, { 0, 0, 255 }, 2);
    cv::line(canvas, p3, p4, { 0, 255, 0 });
    cv::line(canvas, p4, p1, { 0, 255, 0 });
}

inline void
drawCenteredText(cv::Mat& canvas, const std::string& msg, const cv::Point& center)
{
    const auto fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const auto fontScale = 2.0;
    const auto fontThickness = 3;

    int baseline = 0;
    auto textSize = cv::getTextSize(msg, fontFace, fontScale, fontThickness, &baseline);
    baseline += fontThickness;

    cv::Point msgOrigin(center.x - textSize.width / 2, center.y + textSize.height / 2);
    cv::putText(canvas, msg, msgOrigin, fontFace, fontScale, { 0, 255, 0 }, fontThickness);
}

inline cv::Mat
compositeImage(const cv::Mat& background, const cv::Mat& foreground)
{
    std::array<cv::Mat, 4> channels;
    cv::split(foreground, channels);

    cv::Mat foregroundRgb;
    cv::merge(std::array<cv::Mat, 3>{ channels[0], channels[1], channels[2] }, foregroundRgb);

    cv::Mat bg;
    cv::bitwise_and(background, background, bg, channels[3] > 0);

    cv::Mat fg;
    cv::bitwise_and(foregroundRgb, foregroundRgb, fg, channels[3] <= 0);

    cv::Mat out;
    cv::add(bg, fg, out);

    return out;
}

inline cv::Mat
renderWarped(const cv::Mat& canvas, const cv::Mat& overlay, const cv::Mat& transform)
{
    cv::Mat M;
    cv::invert(transform, M);

    cv::Mat warpedOverlay;
    cv::warpPerspective(
      overlay, warpedOverlay, M, cv::Size(canvas.cols, canvas.rows), cv::INTER_AREA, cv::BORDER_TRANSPARENT);

    return compositeImage(canvas, warpedOverlay);
}

inline std::tuple<double, cv::Mat> resizeMaxSideLen(const cv::Mat& input, int maxSideLen) {
    int h = input.rows;
    int w = input.cols;
    double scale = static_cast<double>(maxSideLen) / static_cast<double>(h > w ? h : w);
    cv::Mat resizedImage;
    cv::resize(input, resizedImage, cv::Size(), scale, scale, cv::INTER_AREA);
    return std::make_tuple(scale, resizedImage);
}

#endif // DRAWUTIL_H
