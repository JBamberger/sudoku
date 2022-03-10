#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <cmath>

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

#endif // MATHUTIL_H
