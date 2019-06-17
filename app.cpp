
#include "app.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <iostream>

#include "DigitClassifier.h"
#include "SudokuDetector.h"

using cv::Mat;
using cv::Point;
using cv::Size;
using cv::Vec2f;
using cv::Point2f;
using std::vector;


void drawLine(const cv::Vec2f &line, cv::Mat &img, const cv::Scalar &rgb = CV_RGB(0, 0, 255)) {
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

std::vector<cv::Vec2f>
discardLines(const std::vector<cv::Vec3f> &lines, const float minAngleDiff, const float minRadDiff) {
    std::vector<cv::Vec2f> maxLines;
    for (const auto &line1 : lines) {
        unsigned int wins = 0;
        for (const auto &line2 : lines) {
            float angleDiff = line1[1] - line2[1];
            float radDiff = line1[0] - line2[0];
//            std::cout << angleDiff << " " << radDiff << std::endl;
            if (-minAngleDiff <= angleDiff && angleDiff <= minAngleDiff
                && -minRadDiff <= radDiff && radDiff <= minRadDiff
                && line1[2] < line2[2]) {
                continue;
            }
            wins++;
        }
        if (wins == lines.size()) {
            maxLines.emplace_back(line1[0], line1[1]);
        }
    }
    return maxLines;
}

int main(int argc, char *argv[]) {
    std::cout << "Hello, World!" << std::endl;

    auto sudoku = cv::imread(R"(C:\Users\Jannik\Desktop\sudoku-sk2.jpg)", 0);
    cv::Size newsize = sudoku.size();
    newsize.height = newsize.height / 2;
    newsize.width = newsize.width / 2;
    cv::resize(sudoku, sudoku, newsize);


    cv::Mat blurred;
    cv::Mat thresh;
    cv::Mat threshEdges;
    cv::GaussianBlur(sudoku, blurred, cv::Size(11, 11), 3);
    cv::adaptiveThreshold(blurred, thresh, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
//    cv::Canny(sudoku, edges, 50, 200);
    cv::Canny(thresh, threshEdges, 50, 200);

//    imshow("image", sudoku);
//    imshow("thresh", thresh);
////    imshow("edges", edges);
//    imshow("threshedges", threshEdges);

    std::vector<cv::Vec3f> lines;
    double rho = 1;
    double theta = CV_PI / 180;
    int thr = 200;
    cv::HoughLines(threshEdges, lines, rho, theta, thr);




    float minAngleDiff = 0;
    float minRadDiff = 0;
    auto draw = [&]() {
        cv::Mat sudoku2 = cv::Mat(sudoku);
        cv::cvtColor(sudoku2, sudoku2, CV_GRAY2RGB);
        std::vector<cv::Vec2f> maxLines = discardLines(lines, minAngleDiff, minRadDiff);
        std::cout << minAngleDiff << std::endl;
        for (const auto &line : maxLines) {
            drawLine(line, sudoku2, CV_RGB(0, 255, 0));
        }
        cv::imshow("lines", sudoku2);
    };

    draw();



    using TrackbarAction = std::function<void(int)>;
    TrackbarAction angleChanged = [&](int value) {
        minAngleDiff = ((float) value) * (CV_PI / 180);
        draw();
    };

    TrackbarAction radChanged = [&](int value) {
        minRadDiff = value;
        draw();
    };

    cv::TrackbarCallback trackbarCallback = [](int pos, void *userdata) {
        (*(TrackbarAction *) userdata)(pos);
    };

    int angleVal = 0;
    int radVal = 0;
    cv::createTrackbar("angle", "lines", &angleVal, 360, trackbarCallback, (void *) &angleChanged);
    cv::createTrackbar("radius", "lines", &radVal, 1000, trackbarCallback, (void *) &radChanged);


    cv::waitKey();

//    auto *sudokuDetector = SudokuDetector::createInstance(DetectorType::SIMPLE_DETECTOR);
//    auto *digitClassifier = new DigitClassifier();
//
//    std::vector<cv::Mat> digits = sudokuDetector->detectDigits(sudoku);
//
//    int col = 1;
//    for (const auto& digit : digits) {
//        int number = digitClassifier->classify(digit);
//        printf("%d ", number);
//        if (col % 9 == 0) {
//            printf("\n");
//        }
//        col++;
//    }

    return EXIT_SUCCESS;
}
