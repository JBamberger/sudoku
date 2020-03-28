
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

    cv::Mat sudokuImg = cv::imread(R"(../share/sudoku-sk.jpg)", 0);
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

    for (int i = 0; i < sudoku.cells.size(); i++) {
        // cv::imshow("cell", sudoku.aligned(sudoku.cells.at(i)));
        // cv::waitKey();

        if (digits.at(i) >= 0) {
            cv::Rect2i cell = sudoku.cells.at(i);

            cv::Mat warpedLocation =
              (cv::Mat_<double>(3, 1) << (double)cell.x + cell.width * 0.1, (double)cell.y + cell.height * 0.9, 1.f);
            cv::Mat u = sudoku.unwarpMap * warpedLocation;
            u = u / u.at<double>(2, 0);
            cv::Point2i location{ (int)u.at<double>(0, 0), (int)u.at<double>(1, 0) };

            cv::putText(resultImg,
                        std::to_string(digits.at(i)),
                        location,
                        FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        { 0, 255, 0 },
                        1,
                        cv::LINE_AA);
        }

        // std::cout << digits.at(i) << std::endl;
    }

    Quad c = sudoku.corners;
    cv::line(resultImg, c.tl, c.tr, { 0, 255, 0 });
    cv::line(resultImg, c.tr, c.br, { 0, 255, 0 });
    cv::line(resultImg, c.br, c.bl, { 0, 255, 0 });
    cv::line(resultImg, c.bl, c.tl, { 0, 255, 0 });
    cv::rectangle(resultImg, sudoku.bbox, { 0, 0, 255 });
    cv::imshow("Result", resultImg);
    cv::waitKey();

    return EXIT_SUCCESS;
}
