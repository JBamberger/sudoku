#ifndef UTILS_H
#define UTILS_H

#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
#include <opencv2/core/types.hpp>
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
cv::Point_<T>
contourCenter(const std::vector<cv::Point_<T>>& contour)
{
    cv::Point_<T> center(0, 0);
    for (const auto& point : contour) {
        center.x += point.x;
        center.y += point.y;
    }
    center /= static_cast<T>(contour.size());
    return center;
}

template<typename T>
std::vector<size_t>
argsort(const std::vector<T>& v, std::function<bool(const T&, const T&)> comp)
{

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings when v contains elements of equal values
    std::stable_sort(idx.begin(), idx.end(), [&v, &comp](size_t i1, size_t i2) { return comp(v[i1], v[i2]); });

    return idx;
}

class Quad
{
  public:
    std::array<cv::Point2d, 4> corners;

    friend std::istream& operator>>(std::istream& str, Quad& data);
    friend std::ostream& operator<<(std::ostream& str, const Quad& data);
};

struct SudokuGroundTruth
{
    std::filesystem::path imagePath;
    Quad bounds;

    inline SudokuGroundTruth(std::filesystem::path imagePath, Quad bounds)
      : imagePath(std::move(imagePath))
      , bounds(std::move(bounds))
    {}
};

std::vector<SudokuGroundTruth>
readGroundTruth(const std::filesystem::path& root, const std::filesystem::path& file);

#endif // UTILS_H
