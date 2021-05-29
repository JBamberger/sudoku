#include <SudokuDetection.h>

#include <drawutil.h>
#include <utils.h>

void
SudokuDetection::drawOverlay(cv::Mat& canvas) const
{
    assert(foundSudoku);
    const cv::Scalar CELL_SUDOKU_COLOR = { 0, 255, 0 };
    const cv::Scalar CELL_SOLUTION_COLOR = { 0, 0, 255 };

    drawOrientedRect(canvas, asContour(sudokuCorners));

    cv::Mat M;
    cv::invert(unwarpTransform, M);

    for (int i = 0; i < 81; i++) {

        std::vector<cv::Point> intContour = cellCoords.at(i);
        if (intContour.empty()) {
            continue;
        }

        std::vector<cv::Point2f> floatContour;
        std::transform(std::begin(intContour),
                       std::end(intContour),
                       std::back_inserter(floatContour),
                       [](const auto& p) { return cv::Point2f(p); });

        std::vector<cv::Point2f> contour;
        cv::perspectiveTransform(floatContour, contour, M);

        drawOrientedRect(canvas, contour);

        int cellNum = cellLabels.at(i);
        auto color = CELL_SUDOKU_COLOR;
        if (solution != nullptr) {
            if (cellNum <= 0) {
                color = CELL_SOLUTION_COLOR;
                cellNum = solution->at(i);
            } else {
                assert(cellNum == solution->at(i));
            }
        }

        const auto center = contourCenter(contour);
        const auto msg = std::to_string(cellNum);
        drawCenteredText(canvas, msg, center, color);
    }
}
