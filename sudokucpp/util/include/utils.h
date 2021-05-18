#ifndef UTILS_H
#define UTILS_H

#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <vector>

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
    std::array<cv::Point2d, 4> corners;

    friend std::istream& operator>>(std::istream& str, Quad& data);
    friend std::ostream& operator<<(std::ostream& str, const Quad& data);
};

std::vector<std::pair<std::filesystem::path, Quad>>
readGroundTruth(const std::filesystem::path& file);

inline cv::Point
contourCenter(const std::vector<cv::Point>& contour)
{
    cv::Point center(0, 0);
    for (const auto& point : contour) {
        center.x += point.x;
        center.y += point.y;
    }
    center /= static_cast<int>(contour.size());
    return center;
}

#endif // UTILS_H
