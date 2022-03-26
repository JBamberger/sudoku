#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <cmath>
#include <opencv2/core/types.hpp>

#define M_PI 3.14159265358979323846

template<typename T>
T
orientedAngle(const T& x1, const T& y1, const T& x2, const T& y2)
{
    auto dot = x1 * x2 + y1 * y2;
    auto det = x1 * y2 - y1 * x2;
    auto angle = std::atan2(det, dot);
    return angle;
}

template<typename T>
T
orientedAngle(const cv::Vec<T, 2>& v1, const cv::Vec<T, 2>& v2)
{
    const T dot = v1.x * v2.x + v1.y * v2.y;
    const T det = v1.x * v2.y - v1.y * v2.x;
    return std::atan2(det, dot);
}

#endif // MATHUTIL_H
