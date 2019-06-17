//
// Created by Jannik on 28.05.2019.
//


#include "SudokuDetector.h"
#include <stdexcept>

void drawLine(cv::Vec2f line, cv::Mat &img, cv::Scalar rgb);

std::vector<cv::Mat> SudokuDetectorImpl::detectDigits(cv::Mat &image) {

    cv::imshow("image", image);
    cv::waitKey(0);

    auto outerBox = cv::Mat(image.size(), CV_8UC1);

    cv::GaussianBlur(image, image, cv::Size(11, 11), 0);

    cv::imshow("image", image);
    cv::waitKey(0);

    find_sudoku_blob(image, outerBox);

    cv::imshow("image", outerBox);
    cv::waitKey(0);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(outerBox, lines, 1, CV_PI / 180, 200);

    for (const auto &line : lines) {
        drawLine(line, outerBox, CV_RGB(0, 0, 128));
    }
    cv::imshow("image", outerBox);
    cv::waitKey(0);

    mergeRelatedLines(lines, image);

    SudokuEdges edges = find_sudoku_edges(lines);

    drawLine(edges.top, image, CV_RGB(0, 0, 0));
    drawLine(edges.bottom, image, CV_RGB(0, 0, 0));
    drawLine(edges.left, image, CV_RGB(0, 0, 0));
    drawLine(edges.right, image, CV_RGB(0, 0, 0));

    cv::imshow("image", image);
    cv::waitKey(0);


    cv::Point2f src[4], dst[4];
    int maxLength = find_sudoku_corners(outerBox.size(), edges, src, dst);

    cv::Mat undistorted = cv::Mat(cv::Size(maxLength, maxLength), CV_8UC1);
    cv::warpPerspective(image, undistorted, cv::getPerspectiveTransform(src, dst), cv::Size(maxLength, maxLength));

    cv::imshow("image", undistorted);
    cv::waitKey(0);

    cv::Mat undistortedThreshed = undistorted.clone();
    cv::adaptiveThreshold(undistorted, undistortedThreshed, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV,
                          101, 1);

    cv::imshow("image", undistortedThreshed);
    cv::waitKey(0);

    int dist = ceil((double) maxLength / 9);
    cv::Mat currentCell = cv::Mat(dist, dist, CV_8UC1);

    const int outerCellSize = undistortedThreshed.size().width / 9;
    const int cellOffset = 0;
    const int innerCellSize = outerCellSize - (2 * cellOffset);


    std::vector<cv::Mat> output;
    for (int j = 0; j < 9; j++) {
        for (int i = 0; i < 9; i++) {
            cv::Mat digit;
            undistortedThreshed(cv::Rect(i * outerCellSize + cellOffset, j * outerCellSize + cellOffset, innerCellSize,
                                         innerCellSize)).copyTo(digit);
            output.push_back(digit);
        }
        printf("\n");
    }

    return output;
}

void SudokuDetectorImpl::mergeRelatedLines(std::vector<cv::Vec2f> &lines, cv::Mat &img) {
    std::vector<cv::Vec2f>::iterator current;
    for (current = lines.begin(); current != lines.end(); current++) {
        if ((*current)[0] == 0 && (*current)[1] == -100) continue;
        float p1 = (*current)[0];
        float theta1 = (*current)[1];
        cv::Point pt1current, pt2current;
        if (theta1 > CV_PI * 45 / 180 && theta1 < CV_PI * 135 / 180) {
            pt1current.x = 0;

            pt1current.y = p1 / sin(theta1);

            pt2current.x = img.size().width;
            pt2current.y = -pt2current.x / tan(theta1) + p1 / sin(theta1);
        } else {
            pt1current.y = 0;

            pt1current.x = p1 / cos(theta1);

            pt2current.y = img.size().height;
            pt2current.x = -pt2current.y / tan(theta1) + p1 / cos(theta1);

        }
        std::vector<cv::Vec2f>::iterator pos;
        for (pos = lines.begin(); pos != lines.end(); pos++) {
            if (*current == *pos) continue;
            if (fabs((*pos)[0] - (*current)[0]) < 20 && fabs((*pos)[1] - (*current)[1]) < CV_PI * 10 / 180) {
                float p = (*pos)[0];
                float theta = (*pos)[1];
                cv::Point pt1, pt2;
                if ((*pos)[1] > CV_PI * 45 / 180 && (*pos)[1] < CV_PI * 135 / 180) {
                    pt1.x = 0;
                    pt1.y = p / sin(theta);
                    pt2.x = img.size().width;
                    pt2.y = -pt2.x / tan(theta) + p / sin(theta);
                } else {
                    pt1.y = 0;
                    pt1.x = p / cos(theta);
                    pt2.y = img.size().height;
                    pt2.x = -pt2.y / tan(theta) + p / cos(theta);
                }
                if (((double) (pt1.x - pt1current.x) * (pt1.x - pt1current.x) +
                     (pt1.y - pt1current.y) * (pt1.y - pt1current.y) < 64 * 64) &&
                    ((double) (pt2.x - pt2current.x) * (pt2.x - pt2current.x) +
                     (pt2.y - pt2current.y) * (pt2.y - pt2current.y) < 64 * 64)) {
                    // Merge the two
                    (*current)[0] = ((*current)[0] + (*pos)[0]) / 2;

                    (*current)[1] = ((*current)[1] + (*pos)[1]) / 2;

                    (*pos)[0] = 0;
                    (*pos)[1] = -100;
                }
            }
        }
    }
}

void SudokuDetectorImpl::find_sudoku_blob(cv::Mat &sudoku, cv::Mat &outerBox) {
    cv::adaptiveThreshold(sudoku, outerBox, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 2);
    cv::bitwise_not(outerBox, outerBox);

    cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
    cv::dilate(outerBox, outerBox, kernel);

    int count = 0;
    int max = -1;

    cv::Point maxPt;

    for (int y = 0; y < outerBox.size().height; y++) {
        uchar *row = outerBox.ptr(y);
        for (int x = 0; x < outerBox.size().width; x++) {
            if (row[x] >= 128) {

                int area = cv::floodFill(outerBox, cv::Point(x, y), CV_RGB(0, 0, 64));
                //cv::imshow("image", outerBox);
                //cv::waitKey(0);
                if (area > max) {
                    maxPt = cv::Point(x, y);
                    max = area;
                }
            }
        }
    }

    cv::floodFill(outerBox, maxPt, CV_RGB(255, 255, 255));
    for (int y = 0; y < outerBox.size().height; y++) {
        uchar *row = outerBox.ptr(y);
        for (int x = 0; x < outerBox.size().width; x++) {
            if (row[x] == 64 && x != maxPt.x && y != maxPt.y) {
                int area = floodFill(outerBox, cv::Point(x, y), CV_RGB(0, 0, 0));
            }
        }
    }
    erode(outerBox, outerBox, kernel);
}

int SudokuDetectorImpl::find_sudoku_corners(cv::Size2i imgSize, SudokuEdges &edges, cv::Point2f(&src)[4],
                                            cv::Point2f(&dst)[4]) {
    cv::Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;
    int height = imgSize.height;
    int width = imgSize.width;
    if (edges.left[1] != 0) {
        left1.x = 0;
        left1.y = edges.left[0] / sin(edges.left[1]);
        left2.x = width;
        left2.y = -left2.x / tan(edges.left[1]) + left1.y;
    } else {
        left1.y = 0;
        left1.x = edges.left[0] / cos(edges.left[1]);
        left2.y = height;
        left2.x = left1.x - height * tan(edges.left[1]);
    }

    if (edges.right[1] != 0) {
        right1.x = 0;
        right1.y = edges.right[0] / sin(edges.right[1]);
        right2.x = width;
        right2.y = -right2.x / tan(edges.right[1]) + right1.y;
    } else {
        right1.y = 0;
        right1.x = edges.right[0] / cos(edges.right[1]);
        right2.y = height;
        right2.x = right1.x - height * tan(edges.right[1]);
    }
    bottom1.x = 0;
    bottom1.y = edges.bottom[0] / sin(edges.bottom[1]);
    bottom2.x = width;
    bottom2.y = -bottom2.x / tan(edges.bottom[1]) + bottom1.y;
    top1.x = 0;
    top1.y = edges.top[0] / sin(edges.top[1]);
    top2.x = width;
    top2.y = -top2.x / tan(edges.top[1]) + top1.y;

    // Next, we find the intersection of  these four lines
    double leftA = left2.y - left1.y;
    double leftB = left1.x - left2.x;

    double leftC = leftA * left1.x + leftB * left1.y;

    double rightA = right2.y - right1.y;
    double rightB = right1.x - right2.x;

    double rightC = rightA * right1.x + rightB * right1.y;

    double topA = top2.y - top1.y;
    double topB = top1.x - top2.x;

    double topC = topA * top1.x + topB * top1.y;

    double bottomA = bottom2.y - bottom1.y;
    double bottomB = bottom1.x - bottom2.x;

    double bottomC = bottomA * bottom1.x + bottomB * bottom1.y;

    // Intersection of left and top
    double detTopLeft = leftA * topB - leftB * topA;

    CvPoint ptTopLeft = cvPoint(
            (topB * leftC - leftB * topC) / detTopLeft,
            (leftA * topC - topA * leftC) / detTopLeft
    );

    // Intersection of top and right
    double detTopRight = rightA * topB - rightB * topA;

    CvPoint ptTopRight = cvPoint(
            (topB * rightC - rightB * topC) / detTopRight,
            (rightA * topC - topA * rightC) / detTopRight
    );

    // Intersection of right and bottom
    double detBottomRight = rightA * bottomB - rightB * bottomA;
    CvPoint ptBottomRight = cvPoint(
            (bottomB * rightC - rightB * bottomC) / detBottomRight,
            (rightA * bottomC - bottomA * rightC) / detBottomRight
    );

    // Intersection of bottom and left
    double detBottomLeft = leftA * bottomB - leftB * bottomA;
    CvPoint ptBottomLeft = cvPoint(
            (bottomB * leftC - leftB * bottomC) / detBottomLeft,
            (leftA * bottomC - bottomA * leftC) / detBottomLeft
    );

    int maxLength = (ptBottomLeft.x - ptBottomRight.x) * (ptBottomLeft.x - ptBottomRight.x)
                    + (ptBottomLeft.y - ptBottomRight.y) * (ptBottomLeft.y - ptBottomRight.y);
    int temp = (ptTopRight.x - ptBottomRight.x) * (ptTopRight.x - ptBottomRight.x)
               + (ptTopRight.y - ptBottomRight.y) * (ptTopRight.y - ptBottomRight.y);

    if (temp > maxLength) maxLength = temp;

    temp = (ptTopRight.x - ptTopLeft.x) * (ptTopRight.x - ptTopLeft.x)
           + (ptTopRight.y - ptTopLeft.y) * (ptTopRight.y - ptTopLeft.y);

    if (temp > maxLength) maxLength = temp;

    temp = (ptBottomLeft.x - ptTopLeft.x) * (ptBottomLeft.x - ptTopLeft.x)
           + (ptBottomLeft.y - ptTopLeft.y) * (ptBottomLeft.y - ptTopLeft.y);

    if (temp > maxLength) maxLength = temp;

    maxLength = (int) sqrt((double) maxLength);


    src[0] = ptTopLeft;
    dst[0] = cv::Point2f(0, 0);
    src[1] = ptTopRight;
    dst[1] = cv::Point2f(maxLength - 1, 0);
    src[2] = ptBottomRight;
    dst[2] = cv::Point2f(maxLength - 1, maxLength - 1);
    src[3] = ptBottomLeft;
    dst[3] = cv::Point2f(0, maxLength - 1);

    return maxLength;
}

SudokuEdges SudokuDetectorImpl::find_sudoku_edges(std::vector<cv::Vec2f> &lines) {
    SudokuEdges edges = SudokuEdges();

    double topYIntercept = 100000;
    double topXIntercept = 0;
    double bottomYIntercept = 0, bottomXIntercept = 0;
    double leftXIntercept = 100000, leftYIntercept = 0;
    double rightXIntercept = 0, rightYIntercept = 0;


    for (auto current : lines) {
        float p = current[0];
        float theta = current[1];
        if (p == 0 && theta == -100)
            continue;

        double xIntercept, yIntercept;
        xIntercept = p / cos(theta);
        yIntercept = p / (cos(theta) * sin(theta));
        if (theta > CV_PI * 80 / 180 && theta < CV_PI * 100 / 180) {
            if (p < edges.top[0])
                edges.top = current;

            if (p > edges.bottom[0])
                edges.bottom = current;
        } else if (theta < CV_PI * 10 / 180 || theta > CV_PI * 170 / 180) {
            if (xIntercept > rightXIntercept) {
                edges.right = current;
                rightXIntercept = xIntercept;
            } else if (xIntercept <= leftXIntercept) {
                edges.left = current;
                leftXIntercept = xIntercept;
            }
        }
    }
    return edges;
}


void drawLine(cv::Vec2f line, cv::Mat &img, cv::Scalar rgb = CV_RGB(0, 0, 255)) {
    if (line[1] != 0) {
        float m = -1 / tan(line[1]);
        float c = line[0] / sin(line[1]);

        cv::line(
                img,
                cv::Point(0, c),
                cv::Point(img.size().width, m * img.size().width + c),
                rgb
        );
    } else {
        cv::line(
                img,
                cv::Point(line[0], 0),
                cv::Point(line[0], img.size().height),
                rgb
        );
    }
}

SudokuDetector *SudokuDetector::createInstance(DetectorType type) {
    switch (type) {
        case DetectorType::SIMPLE_DETECTOR:
            return new SudokuDetectorImpl;
        default:
            throw std::invalid_argument("Invalid detector type.");
    }
}

std::vector<cv::Mat> LineBasedDetector::detectDigits(cv::Mat &image) {
    return std::vector<cv::Mat>();
}
