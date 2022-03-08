//
// Copyright 2011 Haoest
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <opencv2/imgproc.hpp>
#include <vector>

// when parsing lines which form a potential sudoku board, consider multiple lines
// the same degree as long as they fall in this tolerance
static double SlopeLineToleranceInDegree = 5;

// only accept images that are titled no more than this angle
static int BoardTiltTolerance = 30;

// if input image is large, try to resize it to this size proportionally. The longer leg between width and height will
// use this size
const double InputImageNormalizeLength = 800;

static int BoardThresholdCandidateSize = 4;
static int BoardThresholdCandidates[] = { 1, 5, 10, 20 };

constexpr size_t HIERARCHY_SIB_NEXT = 0;
constexpr size_t HIERARCHY_SIB_PREV = 1;
constexpr size_t HIERARCHY_CHILD = 2;
constexpr size_t HIERARCHY_PARENT = 3;

float
radiantToDegree(float radiant)
{
    return 180.0 / CV_PI * radiant;
}

float
degreeToRadiant(float degree)
{
    return CV_PI / 180 * degree;
}

void
findExtremas(const std::vector<cv::Point>& contour, int& left, int& right, int& top, int& bottom)
{
    right = bottom = -1;
    left = 1 << 30;
    top = 1 << 30;
    for (const auto& p : contour) {
        left = MIN(left, p.x);
        right = MAX(right, p.x);
        top = MIN(top, p.y);
        bottom = MAX(bottom, p.y);
    }
}

void
findXExtramas(const cv::Mat& mask, int* min, int* max)
{
    int minFound = 0;
    // For each col sum, if sum < width*255 then we find the min
    // then continue to end to search the max, if sum< width*255 then is new max
    for (int i = 0; i < mask.cols; i++) {
        cv::Mat data = mask.col(i);
        auto val = cv::sum(data);
        if (val.val[0] > 0) {
            *max = i;
            if (!minFound) {
                *min = i;
                minFound = 1;
            }
        }
    }
}

void
findYExtremas(const cv::Mat& mask, int* min, int* max)
{
    int minFound = 0;
    // For each col sum, if sum < width*255 then we find the min
    // then continue to end to search the max, if sum< width*255 then is new max
    for (int i = 0; i < mask.rows; i++) {
        cv::Mat data = mask.row(i);
        auto val = cv::sum(data);
        if (val.val[0] > 0) {
            *max = i;
            if (!minFound) {
                *min = i;
                minFound = 1;
            }
        }
    }
}

bool
areLinesSimilarInSlope(float theta1, float theta2)
{
    // theta1 and theta2 are in degrees (as supposed to radiant)
    // return (theta1 > theta2 - SlopeLineToleranceInDegree && theta1 < theta2 + SlopeLineTolerance) || (cv::PI -
    // abs(theta2 - theta1) < SlopeLineTolerance);
    return abs(theta1 - theta2) < SlopeLineToleranceInDegree ||
           180 - MAX(theta1, theta2) + MIN(theta1, theta2) < SlopeLineToleranceInDegree;
}

cv::Rect
findRectFromMask(const cv::Mat& mask)
{
    int xmin, xmax, ymin, ymax;
    xmin = xmax = ymin = ymax = 0;
    findXExtramas(mask, &xmin, &xmax);
    findYExtremas(mask, &ymin, &ymax);

    return { xmin, ymin, xmax - xmin, ymax - ymin };
}

bool
checkLineDistributionByRho(const std::vector<cv::Vec2f>& boardLines, float pivotLineTheta, int boardSpace)
{
    std::vector<int> rhos;
    rhos.reserve(boardLines.size());
    for (const auto& line : boardLines) {
        float rho = line[0];
        float theta = line[1];
        if (areLinesSimilarInSlope(radiantToDegree(theta), pivotLineTheta)) {
            rhos.push_back(rho);
        }
    }
    if (rhos.empty()) {
        return false;
    }
    int firstRho = rhos[0], lastRho = rhos[0];
    for (int i = 1; i < rhos.size(); i++) {
        if (firstRho > rhos[i]) {
            firstRho = rhos[i];
        }
        if (lastRho < rhos[i]) {
            lastRho = rhos[i];
        }
    }
    // normalize so that firstRho is 0;
    int offset = -firstRho;
    for (int& rho : rhos) {
        rho += offset;
    }
    firstRho += offset;
    lastRho += offset;
    if (lastRho < boardSpace / 2) {
        return false;
    }
    int roundingAmount = lastRho / 9 / 10;
    int averageDistance = lastRho / 9;
    int numDistributionMarks = 11;
    int distributionMark[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int rho : rhos) {
        int markIndex = (rho + roundingAmount) / averageDistance;
        distributionMark[markIndex] = 1;
    }
    int matches = 0;
    for (int i = 0; i < numDistributionMarks; i++) {
        matches += distributionMark[i];
    }
    return matches >= 5;
}

void
findLineOccurenceBySlope(std::array<cv::Scalar, 18>& lineSlopeFreq, int freqLineCount, int& highIndex, int& secondIndex)
{
    if (lineSlopeFreq[0].val[0] > lineSlopeFreq[1].val[0]) {
        highIndex = 0;
        secondIndex = 1;
    } else {
        highIndex = 1;
        secondIndex = 0;
    }
    for (int i = 2; i < freqLineCount; i++) {
        int count = lineSlopeFreq[i].val[0];
        if (count >= lineSlopeFreq[highIndex].val[0] && count > lineSlopeFreq[secondIndex].val[0]) {
            secondIndex = highIndex;
            highIndex = i;
        } else if (count > lineSlopeFreq[secondIndex].val[0]) {
            secondIndex = i;
        }
    }
}

cv::Mat
getROIAsImageRef(const cv::Mat& img, int left, int right, int top, int bottom)
{
    return img(cv::Range(top, bottom), cv::Range(left, right));
}

cv::Mat
extractAndSetBoardUpRight(const std::vector<cv::Point>& contours,
                          const cv::Point& contourOffset,
                          const cv::Mat& potentialBoardRoi,
                          float degreeOfHighestCount)
{
    cv::Mat mask_src = cv::Mat::zeros(potentialBoardRoi.rows, potentialBoardRoi.cols, CV_8U);

    std::vector<cv::Point> points;
    points.reserve(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        points.push_back(contours[i]);
        points[i].x -= contourOffset.x;
        points[i].y -= contourOffset.y;
    }
    cv::fillConvexPoly(mask_src, points, cv::Scalar(255));

    cv::Point2f src_center(potentialBoardRoi.cols / 2.0F, potentialBoardRoi.rows / 2.0F);

    double angle = degreeOfHighestCount - 90; // assume horizontal
    if (degreeOfHighestCount < 45) {          // somewhat horizontal
        angle = degreeOfHighestCount;
    } else if (degreeOfHighestCount > 135) { // tilted counter-clockwise, re-orient by rotating clock-wise
        angle = degreeOfHighestCount - 180;
    }
    cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);

    cv::Mat dst;
    cv::warpAffine(potentialBoardRoi, dst, rot_mat, potentialBoardRoi.size());

    cv::Mat mask_dst;
    cv::warpAffine(mask_src, mask_dst, rot_mat, potentialBoardRoi.size());

    cv::Rect boardRoi = findRectFromMask(mask_dst);
    cv::Mat rv(cv::Size(boardRoi.width, boardRoi.height), CV_8U);
    rv.setTo({ 255 });
    dst(boardRoi).copyTo(rv, mask_dst(boardRoi));

    return rv;
}

void
updateLineSlopeFreq(std::array<cv::Scalar, 18>& lineSlopeFreq, int lineCount, float degree)
{
    int bin = degree / (180 / lineCount);
    lineSlopeFreq[bin].val[0] += 1;
    if (lineSlopeFreq[bin].val[1] == 0) {
        lineSlopeFreq[bin].val[1] = degree;
    } else {
        lineSlopeFreq[bin].val[1] = (lineSlopeFreq[bin].val[1] + degree) / 2;
    }
}

cv::Mat
doesResembleBoard(const std::vector<cv::Point>& contours, const cv::Mat& fullSrcGray, const cv::Mat& fullSrcBinary)
{
    cv::Mat rv;

    if (contours.empty()) {
        return rv;
    }

    int left, right, top, bottom;
    findExtremas(contours, left, right, top, bottom);

    ////////////
    // sub image
    cv::Mat potentialBoardRoi = getROIAsImageRef(fullSrcBinary, left, right, top, bottom);
    cv::Mat color_dst;

    cv::Mat roiEdges;
    cv::Canny(potentialBoardRoi, roiEdges, 50, 200);

    std::vector<cv::Vec2f> boardLines;
    int houghThreshold = MIN(potentialBoardRoi.rows, potentialBoardRoi.cols) / 3;
    cv::HoughLines(roiEdges, boardLines, 1, CV_PI / 180, houghThreshold);

    if (boardLines.size() >= 16) { // must have at least 20 lines to form a 9x9 grid, but do allow approximation
        // make 18 bins to hold the lines, so that every 10 degrees get 1 bin
        std::array<cv::Scalar, 18> lineSlopeFreq{};
        for (int i = 0; i < 18; i++) {
            lineSlopeFreq[i] = cv::Scalar(0, 0);
        }
        // also for every CvScalar quadruplet:
        //[0] contains the count of the number of lines that are in the bin
        //[1] contains the average degree of tilt calculated from the degrees of the individual lines
        int freqLineCount = 18;
        for (auto line : boardLines) {
            float theta = line[1];
            float rho = line[0];
            float degree = radiantToDegree(theta);
            updateLineSlopeFreq(lineSlopeFreq, freqLineCount, degree);
        }
        int highIndex = -1, secondIndex = -2;
        findLineOccurenceBySlope(lineSlopeFreq, freqLineCount, highIndex, secondIndex);
        // check to see if lines of highest and second highest counts (in degree) are somewhat perpendicular, and that
        // they are not tilted too badly
        float slopeDifference = abs(lineSlopeFreq[highIndex].val[1] - lineSlopeFreq[secondIndex].val[1]);
        int degreeOfTilt = lineSlopeFreq[highIndex].val[1];
        if (abs(slopeDifference - 90) < SlopeLineToleranceInDegree &&
            (degreeOfTilt < BoardTiltTolerance || degreeOfTilt > 180 - BoardTiltTolerance ||
             (degreeOfTilt > 90 - BoardTiltTolerance && degreeOfTilt < 90 + BoardTiltTolerance))) {
            // pick a side (width or height) as the number to be checked against in rho distribution check
            int boardSpace;
            if (lineSlopeFreq[highIndex].val[0] == lineSlopeFreq[secondIndex].val[0]) {
                boardSpace = MAX(right - left, bottom - top);
            } else {

                boardSpace = lineSlopeFreq[highIndex].val[1] > 45 && lineSlopeFreq[highIndex].val[1] < 135
                               ? bottom - top
                               : right - left;
            }
            bool distributionCheck =
              checkLineDistributionByRho(boardLines, lineSlopeFreq[highIndex].val[1], boardSpace);
            if (distributionCheck) {
                distributionCheck =
                  checkLineDistributionByRho(boardLines, lineSlopeFreq[secondIndex].val[1], boardSpace);
            }
            if (distributionCheck) {
                int padding = (right - left) / 18;
                int roiLeft = MAX(left - padding, 0);
                int roiRight = MIN(right + padding, fullSrcGray.cols - 1);
                int roiTop = MAX(top - padding, 0);
                int roiBottom = MIN(bottom + padding, fullSrcGray.rows - 1);
                cv::Mat potentialBoardNoBorder = getROIAsImageRef(fullSrcGray, roiLeft, roiRight, roiTop, roiBottom);
                rv =
                  extractAndSetBoardUpRight(contours, cv::Point(roiLeft, roiTop), potentialBoardNoBorder, degreeOfTilt);
            }
        }
    }

    return rv;
}

cv::Mat
findBoardFromContour(int curNode,
                     const std::vector<std::vector<cv::Point>>& contours,
                     const std::vector<cv::Vec4i>& hierarchy,
                     const cv::Mat& fullSrcGray,
                     const cv::Mat& fullSrcBinary)
{
    cv::Mat board;
    if (curNode < 0) {
        return board;
    }

    int numSiblings = 0;
    for (int sibling = curNode; sibling >= 0; sibling = hierarchy[sibling][HIERARCHY_SIB_NEXT]) {
        numSiblings++;
        board =
          findBoardFromContour(hierarchy[sibling][HIERARCHY_CHILD], contours, hierarchy, fullSrcGray, fullSrcBinary);
        if (!board.empty()) {
            return board;
        }
    }
    if (72 <= numSiblings && numSiblings <= 90) {
        board = doesResembleBoard(contours[hierarchy[curNode][HIERARCHY_PARENT]], fullSrcGray, fullSrcBinary);
    }
    return board;
}

// if input image is exceedingly big, down size it, otherwise return null
cv::Mat
normalizeSourceImageSize(const cv::Mat& sourceImage)
{
    cv::Mat rv;
    if (MAX(sourceImage.cols, sourceImage.rows) > InputImageNormalizeLength) {
        int normalizedWidth, normalizedHeight;
        if (sourceImage.cols > sourceImage.rows) {
            normalizedWidth = InputImageNormalizeLength;
            normalizedHeight = InputImageNormalizeLength / sourceImage.cols * sourceImage.rows;
        } else {
            normalizedHeight = InputImageNormalizeLength;
            normalizedWidth = InputImageNormalizeLength / sourceImage.rows * sourceImage.cols;
        }
        cv::resize(sourceImage, rv, cv::Size(normalizedWidth, normalizedHeight));
    }

    return rv;
}

cv::Mat
findSudokuBoard(const cv::Mat& fullSrc, int& backgroundThresholdUsed)
{

    cv::Mat fullSrcGray;
    cv::cvtColor(fullSrc, fullSrcGray, cv::COLOR_BGR2GRAY);

    cv::Mat resized = normalizeSourceImageSize(fullSrcGray);
    if (!resized.empty()) {
        fullSrcGray = resized;
    }

    cv::Mat fullSrcGrayBlurred = fullSrcGray.clone();
    cv::Mat rv;
    cv::GaussianBlur(fullSrcGray, fullSrcGrayBlurred, cv::Size(3, 1), 0.95);
    int blockSize = MIN(fullSrcGray.cols, fullSrcGray.rows) / 9 | 1;
    for (int i = 0; i < BoardThresholdCandidateSize && rv.empty(); i++) {
        backgroundThresholdUsed = BoardThresholdCandidates[i];
        cv::Mat fullSrcInverted;
        cv::adaptiveThreshold(fullSrcGrayBlurred,
                              fullSrcInverted,
                              255,
                              cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV,
                              blockSize,
                              backgroundThresholdUsed);

        cv::Mat fullSrcBinary;
        cv::adaptiveThreshold(fullSrcGrayBlurred,
                              fullSrcBinary,
                              255,
                              cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY,
                              blockSize,
                              backgroundThresholdUsed);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(fullSrcInverted, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        rv = findBoardFromContour(0, contours, hierarchy, fullSrcGray, fullSrcBinary);
    }
    return rv;
}

// std::array<cv::Point, 4>
cv::Mat
detect_sudoku_v2(const cv::Mat& imageInput)
{
    int backgroundThresholdMark;
    cv::Mat board = findSudokuBoard(imageInput, backgroundThresholdMark);
    return board;
}
