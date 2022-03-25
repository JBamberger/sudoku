
#include "geometry.h"
#include "mathutil.h"
#include "utils.h"
#include <opencv2/imgproc.hpp>

void
normalizeQuadOrientation(const Contour& contour, Contour& outRect)
{
    auto sortX = argsort<cv::Point>(contour, [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
    auto sortY = argsort<cv::Point>(contour, [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });

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
    Contour rotatedBox;
    rotatedBox.reserve(contour.size());
    for (int i = 0; i < contour.size(); i++) {
        auto index = i + offset;
        if (index >= contour.size()) {
            index -= static_cast<int>(contour.size());
        }
        rotatedBox.push_back(contour.at(index));
    }

    auto delta = rotatedBox.at(1) - rotatedBox.at(0);
    auto angle = orientedAngle<double>(static_cast<double>(delta.x), static_cast<double>(delta.y), 1.0, 0.0);
    angle = angle / M_PI * 180.0;

    outRect.at(0) = rotatedBox.at(0);
    if (angle < -45.0 || 45.0 < angle) {
        for (int i = 1; i < rotatedBox.size(); i++) {
            outRect.at(i) = rotatedBox.at(rotatedBox.size() - i);
        }
    } else {
        for (int i = 1; i < rotatedBox.size(); i++) {
            outRect.at(i) = rotatedBox.at(i);
        }
    }
}

void
approximateQuad(const Contour& contour, Contour& outRect, bool normalizeOrientation)
{
    // compute the distance matrix between all points in the contour and find the two points with
    // max distance
    auto n = contour.size();
    double maxDist = 0.0;
    std::array<size_t, 4> pointIndices{ 0, 0, 0, 0 };
    std::vector<std::vector<double>> distances;
    distances.reserve(n);
    for (int i = 0; i < n; i++) {
        distances.emplace_back(n);
        for (int j = 0; j < n; j++) {
            double dist = cv::norm(contour.at(i) - contour.at(j));
            distances.at(i).at(j) = dist;

            if (dist > maxDist) {
                maxDist = dist;
                pointIndices[0] = i;
                pointIndices[1] = j;
            }
        }
    }

    // Find a third point with max distance to the previous two points
    maxDist = 0;
    std::vector<double> distSum;
    distSum.reserve(n);
    for (int i = 0; i < n; i++) {
        auto s = distances.at(pointIndices[0]).at(i) + distances.at(pointIndices[1]).at(i);
        distSum.push_back(s);
        if (s > maxDist) {
            maxDist = s;
            pointIndices[2] = i;
        }
    }

    // find a fourth point with the max distance to the previous three points
    maxDist = 0;
    for (int i = 0; i < n; i++) {
        auto s = distSum.at(i) + distances.at(pointIndices[2]).at(i);
        if (s > maxDist) {
            maxDist = s;
            pointIndices[3] = i;
        }
    }

    // Put the points in the same order as they appear in the contour
    std::sort(std::begin(pointIndices), std::end(pointIndices));

    // Resolve the point indices to actual points and put them into the output vector
    outRect.clear();
    for (size_t pointIndex : pointIndices) {
        outRect.push_back(contour.at(pointIndex));
    }

    // If desired, ensure that the upper left corner of the rectangle is the first point.
    if (normalizeOrientation) {
        normalizeQuadOrientation(outRect, outRect);
    }
}

double
angleCos(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2)
{
    cv::Point d1 = p0 - p1;
    cv::Point d2 = p2 - p1;

    // (d1*d2) / sqrt(|d1|^2 * |d2|^2)
    // (d1*d2) /     (|d1|   * |d2|  )

    auto num = d1.x * d2.x + d1.y * d2.y;
    auto denom = (d1.x * d1.x + d1.y * d1.y) * (d2.x * d2.x + d2.y * d2.y);

    return num / std::sqrt(denom);
}

void
convexContourNms(const std::vector<Contour>& contours,
                 const std::vector<double>& scores,
                 std::vector<Contour>& output,
                 double iouThresh)
{
    assert(contours.size() == scores.size());

    output.reserve(contours.size());

    auto indices = argsort<double>(scores, [](const auto& a, const auto& b) { return a > b; });

    for (auto index : indices) {
        const auto& contour = contours[index];

        bool keep = true;
        for (const auto& keptContour : output) {
            if (convexContourIoU(contour, keptContour) >= iouThresh) {
                keep = false;
                break;
            }
        }

        if (keep) {
            output.push_back(contour);
        }
    }
}

std::vector<Contour>
convexContourAreaNms(const std::vector<Contour>& contours, double iouThresh)
{
    std::vector<double> areas;
    areas.resize(contours.size());
    std::transform(contours.begin(), contours.end(), areas.begin(), [](const auto& c) { return cv::contourArea(c); });

    std::vector<Contour> output;
    convexContourNms(contours, areas, output, iouThresh);
    return output;
}

double
convexContourIoU(const Contour& a, const Contour& b)
{
    std::vector<cv::Point2f> interOut;
    const double inter = cv::intersectConvexConvex(a, b, interOut, true);
    const double a1 = cv::contourArea(a, false);
    const double a2 = cv::contourArea(b, false);

    return inter / (a1 + a2 - inter);
}