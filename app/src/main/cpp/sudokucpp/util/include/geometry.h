
#ifndef SUDOKU4ANDROID_GEOMETRY_H
#define SUDOKU4ANDROID_GEOMETRY_H

#include "mathutil.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/core/types.hpp>

//############################################################################
// TYPES
//############################################################################

using Contour = std::vector<cv::Point>;

template<typename T>
class Quad_
{
  public:
    std::array<cv::Point_<T>, 4> corners;

    explicit Quad_();
    explicit Quad_(cv::Point_<T> a, cv::Point_<T> b, cv::Point_<T> c, cv::Point_<T> d);

    void normalizeOrientation();
    void operator*=(T scale);
    void operator+=(cv::Vec<T, 2> displacement);

    [[nodiscard]] cv::Point_<T> center() const;
    [[nodiscard]] T area() const;

    template<typename>
    inline friend std::istream& operator>>(std::istream& str, Quad_<T>& data);
    template<typename>
    inline friend std::ostream& operator<<(std::ostream& str, const Quad_<T>& data);
};

typedef Quad_<double> Quad;

//############################################################################
// GENERAL-PURPOSE FUNCTIONS
//############################################################################

template<typename T, size_t N>
Contour
asContour(const std::array<cv::Point_<T>, N>& points);

template<typename T>
cv::Point_<T>
contourCenter(const std::vector<cv::Point_<T>>& contour);

void
normalizeQuadOrientation(const Contour& contour, Contour& outRect);

void
approximateQuad(const Contour& contour, Contour& outRect, bool normalizeOrientation = true);

/**
 * Computes the cosine of the angle enclosed by the vectors p1p0 and p1p2.
 *
 *
 * @param p0 First point.
 * @param p1 Second point (The angle is measured at this point).
 * @param p2 Third point.
 * @return Cosine of the angle enclosed at p1.
 */
double
angleCos(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2);

/**
 * Performs non-maximum suppression on a number of convex contours. NMS sorts the shapes by the provided scores and
 * discards all shapes apart from the highest-scoring one that overlap with an IoU higher than the provided threshold.
 *
 * @param contours Input contours, must be convex.
 * @param scores Scores for each of the contour. Must have the same size as contours.
 * @param output Container where the de-duplicated outputs are placed.
 * @param iouThresh Threshold for IoU comparison.
 */
void
convexContourNms(const std::vector<Contour>& contours,
                 const std::vector<double>& scores,
                 std::vector<Contour>& output,
                 double iouThresh);

/**
 * Performs NMS based on the shape area.
 *
 * @param contours Input shapes.
 * @param iouThresh Threshold for IoU comparison.
 * @return Vector of de-duplicated input shapes.
 *
 * @see convexContourNms
 */
std::vector<Contour>
convexContourAreaNms(const std::vector<Contour>& contours, double iouThresh);

/**
 * Computes the Intersection over Union (IoU) of two convex contours. The function does not check werther the contours
 * are actually convex. If they aren't the results might be wrong.
 *
 * @param a First convex contour.
 * @param b Second convex contour.
 * @return Intersection over Union between the contours.
 */
double
convexContourIoU(const Contour& a, const Contour& b);

//############################################################################
// IMPLEMENTATIONS
//############################################################################

//----------------------------------------------------------------------------
// Quad
//----------------------------------------------------------------------------

template<typename T>
Quad_<T>::Quad_()
  : corners{}
{}

template<typename T>
inline std::istream&
operator>>(std::istream& str, Quad_<T>& data)
{
    char delim;
    Quad tmp;
    if (str >> tmp.corners.at(0).x >> delim >> tmp.corners.at(0).y >> delim //
        >> tmp.corners.at(1).x >> delim >> tmp.corners.at(1).y >> delim     //
        >> tmp.corners.at(2).x >> delim >> tmp.corners.at(2).y >> delim     //
        >> tmp.corners.at(3).x >> delim >> tmp.corners.at(3).y) {

        data = tmp;
    } else {
        str.setstate(std::ios::failbit);
    }
    return str;
}

template<typename T>
inline std::ostream&
operator<<(std::ostream& str, const Quad_<T>& data)
{
    std::string delim = ", ";
    str << data.corners.at(0).x << delim << data.corners.at(0).y << delim //
        << data.corners.at(1).x << delim << data.corners.at(1).y << delim //
        << data.corners.at(2).x << delim << data.corners.at(2).y << delim //
        << data.corners.at(3).x << delim << data.corners.at(3).y;
    return str;
}

template<typename T>
void
Quad_<T>::normalizeOrientation()
{
    auto sortX = argsort(
      corners.cbegin(), corners.cend(), [](const cv::Point_<T>& a, const cv::Point_<T>& b) { return a.x < b.x; });
    auto sortY = argsort(
      corners.cbegin(), corners.cend(), [](const cv::Point_<T>& a, const cv::Point_<T>& b) { return a.y < b.y; });

    std::array<size_t, 4> bins{ 0, 0, 0, 0 };
    bins[sortX[0]] += 1;
    bins[sortX[1]] += 1;
    bins[sortY[0]] += 1;
    bins[sortY[1]] += 1;

    int offset = 0;
    for (int i = 0; i < 4; i++) {
        if (bins[i] >= 2) {
            offset = i;
            break;
        }
    }

    const auto angleDeg = orientedAngle(corners[(offset + 1) % 4] - corners[offset], cv::Vec2d(1.0, 0.0));
    const auto reversed = angleDeg < -M_PI / 4 || M_PI / 4 < angleDeg;

    if (offset == 0 && !reversed) {
        return;
    }

    std::array<cv::Point_<T>, 4> cpy;
    std::copy(corners.cbegin(), corners.cend(), cpy.begin());

    if (reversed) {
        for (int i2 = 0; i2 < 4; i2++) {
            corners[i2] = cpy[(offset - i2) % 4];
        }
    } else {
        for (int i1 = 0; i1 < 4; i1++) {
            corners[i1] = cpy[(offset + i1) % 4];
        }
    }
}

template<typename T>
void
Quad_<T>::operator*=(T scale)
{
    for (auto& p : corners) {
        p *= scale;
    }
}

template<typename T>
void
Quad_<T>::operator+=(cv::Vec<T, 2> displacement)
{
    for (auto& p : corners) {
        p += displacement;
    }
}

template<typename T>
cv::Point_<T>
Quad_<T>::center() const
{
    return std::accumulate(corners.cbegin(), corners.cend(), cv::Point_<T>()) / 4;
}

template<typename T>
T
Quad_<T>::area() const
{
    /*
     *       a
     *   0 -----> 1
     *   |      / ^
     * b |    /   | c
     *   v  /     |
     *   3 <----- 2
     *        d
     */

    const auto a = corners[1] - corners[0];
    const auto b = corners[3] - corners[0];
    const auto c = corners[1] - corners[2];
    const auto d = corners[3] - corners[2];

    return 0.5 * (std::abs(a.x * b.y - b.x * a.y) + std::abs(c.x * d.y - d.x * c.y));
}

template<typename T>
Quad_<T>::Quad_(cv::Point_<T> a, cv::Point_<T> b, cv::Point_<T> c, cv::Point_<T> d)
  : corners{ a, b, c, d }
{}

//----------------------------------------------------------------------------
// General-purpose functions
//----------------------------------------------------------------------------

template<typename T, size_t N>
Contour
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

#endif // SUDOKU4ANDROID_GEOMETRY_H
