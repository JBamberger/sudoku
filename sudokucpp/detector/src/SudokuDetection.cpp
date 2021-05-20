#include <SudokuDetection.h>

#include <drawutil.h>
#include <utils.h>

void
SudokuDetection::drawOverlay(cv::Mat& canvas) const
{
    assert(foundSudoku);
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

        const auto msg = std::to_string(cellLabels.at(i));
        const auto center = contourCenter(contour);
        drawCenteredText(canvas, msg, center);
    }
}
