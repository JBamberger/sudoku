
#include "DigitClassifier.h"
#include "SudokuDetector.h"

#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

int
main(int argc, char* argv[])
{

    cv::Mat sudokuImg = cv::imread(R"(../share/sudoku-sk.jpg)", IMREAD_GRAYSCALE );
    cv::resize(sudokuImg, sudokuImg, cv::Size(), 0.5, 0.5);

    DigitClassifier classifier = DigitClassifier();
    Sudoku sudoku = SudokuDetector().detect_sudoku(sudokuImg);

    std::vector<int> digits;
    digits.reserve(81);
    for (const auto& cell : sudoku.cell_contents) {
        digits.push_back(cv::countNonZero(cell) > 0.08 * 20 * 20 ? classifier.classify(cell) : -1);
    }

    cv::Mat resultImg;
    cv::cvtColor(sudokuImg, resultImg, COLOR_GRAY2BGR);

    auto unwarp = [&sudoku](const cv::Point2d& p) {
        cv::Mat warpedLocation = (cv::Mat_<double>(3, 1) << p.x, p.y, 1.f);
        cv::Mat u = sudoku.unwarpMap * warpedLocation;
        u = u / u.at<double>(2, 0);
        return cv::Point2i(static_cast<int>(u.at<double>(0, 0)), static_cast<int>(u.at<double>(1, 0)));
    };

    for (int i = 0; i < sudoku.cells.size(); i++) {
        cv::Rect2i cell = sudoku.cells.at(i);

        std::vector<cv::Point> pts = {
            unwarp({ static_cast<double>(cell.x), static_cast<double>(cell.y) }),
            unwarp({ static_cast<double>(cell.x + cell.width), static_cast<double>(cell.y) }),
            unwarp({ static_cast<double>(cell.x + cell.width), static_cast<double>(cell.y + cell.height) }),
            unwarp({ static_cast<double>(cell.x), static_cast<double>(cell.y + cell.height) })
        };
        cv::polylines(resultImg, pts, true, { 255, 0, 0 });


        if (digits.at(i) >= 0) {
            cv::Point2i location = unwarp({ (double)cell.x + cell.width * 0.1, (double)cell.y + cell.height * 0.9 });

            cv::putText(resultImg,
                        std::to_string(digits.at(i)),
                        location,
                        FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        { 0, 255, 0 },
                        1,
                        cv::LINE_AA);
        }
    }

    Quad c = sudoku.corners;
    std::vector<cv::Point> pts = { c.tl, c.tr, c.br, c.bl };
    cv::polylines(resultImg, pts, true, { 0, 255, 0 });
    cv::rectangle(resultImg, sudoku.bbox, { 0, 0, 255 });
    cv::imshow("Result", resultImg);
    cv::waitKey();

    return EXIT_SUCCESS;
}
